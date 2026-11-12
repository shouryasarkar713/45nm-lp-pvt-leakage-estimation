"""Physics-structured multi-head MLP for one transistor case.
   id = id0(common-mode) + sgn(Vds)*|Vds|*G ;  ig, ib : sign x 10^(log10|I|) ;  is = -id-ig-ib  (KCL by construction)
   Heads: log10|id0|, log10(|R|/|vds|) with R = id-id0, log10|ig|, log10|ib| plus sign logits."""
import numpy as np, pandas as pd, torch, torch.nn as nn, time
from ml_data import *
from ml_infer import StructInference, HEADS
import os; torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 16)))

def targets(d):
    """Training targets (need the zero-Vds twin columns id0,ig0,is0,ib0)."""
    vds = (d.vd - d.vs).values; R = d["id"].values - d["id0"].values
    ok = np.abs(vds) > 1e-12
    T = {"id0": lg(d["id0"].values), "g": np.where(ok, lg(R) - lg(np.where(ok, vds, 1.0)), 0.0),
         "ig": lg(d["ig"].values), "ib": lg(d["ib"].values)}
    S = {"id0": (d["id0"].values >= 0).astype(float), "g": np.where(ok, (np.sign(R) * np.sign(vds) >= 0).astype(float), 1.0),
         "ig": (d["ig"].values >= 0).astype(float), "ib": (d["ib"].values >= 0).astype(float)}
    M = {"id0": np.ones(len(d), bool), "g": ok, "ig": np.ones(len(d), bool), "ib": np.ones(len(d), bool)}
    return T, S, M

class Net(nn.Module):
    def __init__(self, nin, width=256, depth=5):
        super().__init__(); L = []; n = nin
        for _ in range(depth): L += [nn.Linear(n, width), nn.SiLU()]; n = width
        self.trunk = nn.Sequential(*L); self.reg = nn.Linear(width, 4); self.sg = nn.Linear(width, 4)
    def forward(self, x): h = self.trunk(x); return self.reg(h), self.sg(h)

class StructModel:
    def __init__(self, case, width=256, depth=5): self.case = case; self.width = width; self.depth = depth
    def _prep(self, d): return features(d, self.case).values.astype(np.float32)
    def fit(self, tr, va_df=None, epochs=60, bs=2048, lr=2e-3, log=print, seed=SEEDS["ml"], K=1):
        self.nets = []
        for k in range(K):
            log(f"  -- ensemble member {k+1}/{K}"); self._fit_one(tr, va_df, epochs, bs, lr, log, seed + 17 * k, first=(k == 0))
        return self
    def _fit_one(self, tr, va_df, epochs, bs, lr, log, seed, first):
        torch.manual_seed(seed); np.random.seed(seed)
        X = self._prep(tr); T, S, M = targets(tr)
        if first:
            self.mu = X.mean(0); self.sd = X.std(0) + 1e-9
            self.tmu = {h: T[h][M[h]].mean() for h in HEADS}; self.tsd = {h: T[h][M[h]].std() + 1e-9 for h in HEADS}
            self.sconst = {}
            for h in HEADS:
                p = S[h][M[h]].mean(); self.sconst[h] = (1.0 if p > 0.5 else 0.0) if min(p, 1 - p) < 1e-4 else None
        Xn = torch.tensor((X - self.mu) / self.sd)
        Tn = torch.tensor(np.stack([(T[h] - self.tmu[h]) / self.tsd[h] for h in HEADS], 1), dtype=torch.float32)
        Sn = torch.tensor(np.stack([S[h] for h in HEADS], 1), dtype=torch.float32); Mn = torch.tensor(np.stack([M[h] for h in HEADS], 1))
        sw = torch.tensor([0.0 if self.sconst[h] is not None else 1.0 for h in HEADS])
        net = Net(X.shape[1], self.width, self.depth); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-6)
        n = len(Xn); steps = epochs * (n // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        best = (1e9, None); t0 = time.time()
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n)
            for i in range(n // bs):
                idx = perm[i * bs:(i + 1) * bs]; r, s = net(Xn[idx]); m = Mn[idx].float()
                lreg = (nn.functional.smooth_l1_loss(r, Tn[idx], reduction="none", beta=0.05) * m).sum() / m.sum()
                lsg = ((nn.functional.binary_cross_entropy_with_logits(s, Sn[idx], reduction="none") * m) * sw).sum() / (m * sw).sum().clamp(min=1)
                loss = lreg + 0.1 * lsg; opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if va_df is not None and (ep % 5 == 4 or ep == epochs - 1):
                self.nets.append(net); pv = self.predict(va_df); self.nets.pop(); mm = metrics(va_df["id"], pv["id"])["med_err_pct"]
                log(f"  ep{ep+1:3d} loss {loss.item():.4f} val id med {mm:.3f}%  ({time.time()-t0:.0f}s)")
        self.nets.append(net)
    def predict(self, d):
        X = (self._prep(d) - self.mu) / self.sd
        with torch.no_grad():
            outs = [n(torch.tensor(X)) for n in self.nets]
        r = np.mean([o[0].numpy() for o in outs], 0); s = np.mean([o[1].numpy() for o in outs], 0)
        out = {}
        for k, h in enumerate(HEADS):
            mag = 10 ** (r[:, k] * self.tsd[h] + self.tmu[h])
            pos = (s[:, k] >= 0) if self.sconst[h] is None else np.full(len(d), self.sconst[h] > 0.5)
            out[h] = (mag, np.where(pos, 1.0, -1.0))
        vds = (d.vd - d.vs).values
        R = np.sign(vds) * out["g"][1] * out["g"][0] * np.abs(vds)
        idp = out["id0"][0] * out["id0"][1] + R; igp = out["ig"][0] * out["ig"][1]; ibp = out["ib"][0] * out["ib"][1]
        return pd.DataFrame({"id": idp, "ig": igp, "is": -idp - igp - ibp, "ib": ibp}, index=d.index)
    def export(self):
        nets = [dict(W=[(m.weight.detach().numpy(), m.bias.detach().numpy()) for m in list(n.trunk) if isinstance(m, nn.Linear)],
                     Wr=(n.reg.weight.detach().numpy(), n.reg.bias.detach().numpy()), Ws=(n.sg.weight.detach().numpy(), n.sg.bias.detach().numpy())) for n in self.nets]
        return dict(case=self.case, mu=self.mu, sd=self.sd, tmu=self.tmu, tsd=self.tsd, sconst=self.sconst, nets=nets)

