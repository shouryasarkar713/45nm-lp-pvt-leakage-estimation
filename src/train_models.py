"""Part A.2 - train/compare models per case, select on VALIDATION, then evaluate the selected model on TEST.
Usage: train_models.py <stage> [cases]   stage = fit  (train all candidates, write validation table + selection)
                                          test = evaluate the selected models on the held-out test split (once)."""
import sys, os, json, time, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import *
from ml_data import *
from ml_baseline import TerminalModel
from ml_infer import StructInference

MD = os.path.join(OUT, "models"); TD = os.path.join(OUT, "tables"); os.makedirs(MD, exist_ok=True); os.makedirs(TD, exist_ok=True)
FAMILIES = ["Ridge", "RandomForest", "HistGradBoost", "MLP"]
STRUCT_EPOCHS = int(os.environ.get("STRUCT_EPOCHS", 60))

def term_rows(case, name, data, y, p):
    rows = []; agg = []
    for t in TERM:
        m = metrics(y[t].values, p[t].values); rows.append(dict(case=case, model=name, data=data, terminal=t, **m)); agg.append(m)
    mean = {k: float(np.mean([a[k] for a in agg])) for k in agg[0]}
    return rows, mean

def fit_case(case):
    from ml_struct import StructModel   # torch only needed for training
    t0 = time.time(); tr = load_train(case, extra=True); va, te = load_val_test(case)
    spec = tr[tr.src == "spec"]; Xv = features(va, case); log = lambda m: print(f"[{case}] {m}", flush=True)
    log(f"train rows: spec {len(spec)}, extended {len(tr)}, val {len(va)}")
    rows = []; summary = []; cands = {}
    base_ext = tr.sample(n=min(len(tr), 80000), random_state=SEEDS["ml"])   # compute-bounded subsample for the four classical families
    for data, T in (("spec7000", spec), ("extended", base_ext)):
        X = features(T, case)
        for fam in FAMILIES:
            if data == "spec7000" and fam == "RandomForest" and False: continue
            m = TerminalModel(fam).fit(X, T); p = m.predict(Xv)
            r, mean = term_rows(case, fam, data, va, p); rows += r; summary.append(dict(case=case, model=fam, data=data, **mean))
            cands[(fam, data)] = m; log(f"{fam:14s} {data:9s} val medErr {mean['med_err_pct']:.2f}%  p95 {mean['p95_err_pct']:.1f}%  R2log {mean['r2_log']:.4f}  ({time.time()-t0:.0f}s)")
    sm = StructModel(case).fit(tr, va, epochs=STRUCT_EPOCHS, log=log, K=int(os.environ.get('STRUCT_K', 3))); p = sm.predict(va)
    r, mean = term_rows(case, "StructMLP", "extended", va, p); rows += r; summary.append(dict(case=case, model="StructMLP", data="extended", **mean))
    cands[("StructMLP", "extended")] = sm; log(f"StructMLP extended val medErr {mean['med_err_pct']:.2f}% p95 {mean['p95_err_pct']:.1f}% ({time.time()-t0:.0f}s)")
    S = pd.DataFrame(summary); ext = S[S.data == "extended"]
    best = ext.loc[ext.med_err_pct.idxmin()]            # selection criterion: mean over terminals of validation median relative error
    obj = cands[(best.model, "extended")]
    payload = obj.export() if best.model == "StructMLP" else obj
    joblib.dump(dict(case=case, kind=best.model, model=payload, train_rows=len(tr)), os.path.join(MD, f"model_{case}.joblib"), compress=3)
    joblib.dump(dict(case=case, kind="StructMLP", model=sm.export()), os.path.join(MD, f"struct_{case}.joblib"), compress=3)
    pd.DataFrame(rows).to_csv(os.path.join(TD, f"val_terminals_{case}.csv"), index=False); S.to_csv(os.path.join(TD, f"val_summary_{case}.csv"), index=False)
    json.dump(dict(case=case, chosen=best.model, val_med_err_pct=float(best.med_err_pct)), open(os.path.join(TD, f"selection_{case}.json"), "w"))
    log(f"SELECTED {best.model} (val medErr {best.med_err_pct:.3f}%)")

def load_chosen(case):
    d = joblib.load(os.path.join(MD, f"model_{case}.joblib"))
    return StructInference(d["model"]) if d["kind"] == "StructMLP" else d["model"], d["kind"]

def test_case(case):
    m, kind = load_chosen(case); va, te = load_val_test(case)
    p = m.predict(te) if kind == "StructMLP" else m.predict(features(te, case))
    rows, mean = term_rows(case, kind, "test", te, p)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for a, t in zip(ax, TERM):
        y = te[t].values; q = p[t].values; ok = np.sign(y) == np.sign(q)
        a.loglog(np.abs(y)[ok] + 1e-30, np.abs(q)[ok] + 1e-30, ".", ms=2, alpha=.4, label="sign match")
        a.loglog(np.abs(y)[~ok] + 1e-30, np.abs(q)[~ok] + 1e-30, "r.", ms=3, label="sign mismatch")
        lo, hi = 1e-22, np.abs(y).max() * 2; a.plot([lo, hi], [lo, hi], "k--", lw=.7); a.set_xlim(1e-20, hi); a.set_ylim(1e-20, hi)
        a.set_title(f"{case} {t}"); a.set_xlabel("SPICE |I| (A)"); a.set_ylabel("ML |I| (A)"); a.grid(alpha=.3)
    ax[0].legend(fontsize=7); plt.tight_layout(); os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    plt.savefig(os.path.join(OUT, "plots", f"pred_vs_spice_{case}.png"), dpi=130); plt.close()
    return rows, dict(case=case, model=kind, **mean)

if __name__ == "__main__":
    stage = sys.argv[1]; cases = sys.argv[2].split(",") if len(sys.argv) > 2 else CASES
    if stage == "fit":
        for c in cases: fit_case(c)
    elif stage == "test":
        R = []; S = []
        for c in CASES: r, s = test_case(c); R += r; S.append(s)
        pd.DataFrame(R).to_csv(os.path.join(TD, "test_terminals.csv"), index=False); pd.DataFrame(S).to_csv(os.path.join(TD, "test_summary.csv"), index=False)
        print(pd.DataFrame(S).to_string())
