"""Additional held-out TEST statistics computed from the existing chosen models' predictions (no retraining, no selection)."""
import os, numpy as np, pandas as pd
from config import *
from ml_data import *
from train_models import load_chosen
def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
if __name__ == "__main__":
    rows = []
    for c in CASES:
        m, kind = load_chosen(c); va, te = load_val_test(c); p = m.predict(te) if kind == "StructMLP" else m.predict(features(te, c))
        for t in TERM:
            y = te[t].values; q = p[t].values; rel = np.abs(q - y) / np.maximum(np.abs(y), I_FLOOR)
            rows.append(dict(case=c, model=kind, terminal=t, r2_signed=r2(y, q), r2_log10_floor1e30=r2(lg(y, 1e-30), lg(q, 1e-30)), r2_log10_floor1e15=r2(lg(y, 1e-15), lg(q, 1e-15)),
                             mae_A=np.mean(np.abs(q - y)), rmse_A=np.sqrt(np.mean((q - y) ** 2)), median_rel_err_pct=100 * np.median(rel), p95_rel_err_pct=100 * np.quantile(rel, .95),
                             sign_acc_pct=100 * np.mean((np.sign(q) == np.sign(y)) | (np.abs(y) < I_FLOOR))))
        Y = te[TERM].values; Q = p[TERM].values; k = lambda A: np.abs(A.sum(1)) / np.maximum(np.abs(A).max(1), I_FLOOR)
        rows.append(dict(case=c, model=kind, terminal="KCL_sum", median_rel_err_pct=100 * np.median(k(Q)), p95_rel_err_pct=100 * np.quantile(k(Q), .95), mae_A=np.mean(np.abs(Q.sum(1))),
                         rmse_A=np.nan, r2_signed=np.nan, note=f"|sum of 4 predicted currents|/max|I| ; SPICE labels: median {100*np.median(k(Y)):.2e} %, p95 {100*np.quantile(k(Y),.95):.2e} %"))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "tables", "test_terminals_extended.csv"), index=False); print(d.round(4).to_string())
