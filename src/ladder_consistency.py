"""Do relaxed-tolerance rungs change the answer where the strict (assignment) options also converge?"""
import numpy as np, pandas as pd
from multiprocessing import Pool
from gen_gate import *
def f(a):
    kind, ab, ci, rung = a; C = load_corners("official"); c = C.iloc[ci]; nm, pm = corner_procs(c)
    p, devs, inputs, nodes, conv, out = sim_ref(kind, ab, c.vdd, c.temp, nm, pm, rung); p["_in_a"] = c.vdd * ab[0]; p["_in_b"] = c.vdd * ab[1] if len(ab) > 1 else 0
    w = kcl_check(devs, p, nm, pm, c.temp, c.vdd)[0] if "v(out)" in p else np.inf
    return p.get("i(vdd)", np.nan), w, p.get("v(out)", np.nan)
if __name__ == "__main__":
    off = pd.read_csv("../data/gate_spice_official.csv"); s = off[(off.attempt == 0)].sample(300, random_state=1)
    jobs = [(r.kind, (int(r.a), int(r.b)) if r.b >= 0 else (int(r.a),), int(r.corner), k) for r in s.itertuples() for k in (0, 3, 4)]
    with Pool(8) as p: res = p.map(f, jobs)
    R = np.array(res).reshape(-1, 3, 3); I = R[:, :, 0]; W = R[:, :, 1]; V = R[:, :, 2]
    out = []
    for k, rung in ((1, 3), (2, 4)):
        valid = (W[:, 0] <= KCL_TOL) & (W[:, k] <= KCL_TOL); d = np.abs(I[valid, k] / I[valid, 0] - 1)
        dv = np.abs(V[valid, k] - V[valid, 0])
        out.append(dict(rung=rung, n=int(valid.sum()), median_rel_diff=float(np.median(d)), p95_rel_diff=float(np.quantile(d, .95)), p99_rel_diff=float(np.quantile(d, .99)), max_rel_diff=float(d.max()),
                        median_dVout_V=float(np.median(dv)), p95_dVout_V=float(np.quantile(dv, .95)), max_dVout_V=float(dv.max())))
        print(out[-1])
    import os; os.makedirs("../out/audit", exist_ok=True); pd.DataFrame(out).to_csv("../out/audit/ladder_vs_strict.csv", index=False)
