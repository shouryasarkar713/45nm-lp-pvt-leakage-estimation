"""Development-corner ablation (200 development corners, seed 777, different from the official corners)."""
import os, numpy as np, pandas as pd
from config import *
from corners import load_corners
from gate_eval import predict_gates, accuracy
from ml_data import *
from ml_baseline import TerminalModel
from train_models import load_chosen

def row(name, g):
    r = []
    for q, (ref, ml) in {"T": ("i_tot_vdd", "ml_T"), "S": ("i_vdd", "ml_S")}.items():
        a = accuracy(g[ref].values, g[ml].values); r += [a["mean_acc"], a["med_err"], a["p95_err"]]
    return dict(model=name, T_mean_acc=r[0], T_median_err=r[1], T_p95_err=r[2], S_mean_acc=r[3], S_median_err=r[4], S_p95_err=r[5])

if __name__ == "__main__":
    g = pd.read_csv(os.path.join(DATA, "gate_spice_dev.csv")); g = g[g.ok == 1]; C = load_corners("dev"); rows = []
    # classical best (Histogram Gradient Boosting) on the extended 80k subsample
    H = {}
    for c in CASES:
        tr = load_train(c).sample(n=80000, random_state=SEEDS["ml"]) if True else None
        H[c] = TerminalModel("HistGradBoost").fit(features(tr, c), tr)
    gg, _ = predict_gates(g, C, lambda case, df: H[case].predict(features(df, case))); rows.append(row("HistGradBoost (80k extended rows)", gg))
    M = {c: load_chosen(c) for c in CASES}
    gg, _ = predict_gates(g, C, lambda case, df: M[case][0].predict(df) if M[case][1] == "StructMLP" else M[case][0].predict(features(df, case))); rows.append(row("Final selected models (structured MLP ensemble)", gg))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "dev_ablation.csv"), index=False); print(pd.DataFrame(rows).round(3).to_string())
