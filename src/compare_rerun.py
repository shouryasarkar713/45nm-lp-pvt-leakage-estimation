"""Compares a full end-to-end re-execution (run_all.sh in another directory) with the stored results.  Usage: compare_rerun.py <rerun_root>"""
import sys, os, json, glob, numpy as np, pandas as pd
from config import *
def q(x): x = np.asarray(x, float); x = x[np.isfinite(x)]; return dict(median=float(np.median(x)), p95=float(np.quantile(x, .95)), max=float(x.max()), n=int(len(x)))
if __name__ == "__main__":
    R = sys.argv[1]; out = {}
    for c in CASES:
        a = pd.read_csv(os.path.join(R, "data", "base", f"{c}.csv")); b = pd.read_csv(os.path.join(DATA, "base", f"{c}.csv")); A = a[["id", "ig", "is", "ib"]].values; B = b[["id", "ig", "is", "ib"]].values; m = np.abs(B) > 1e-15
        out[f"base_{c}"] = dict(inputs_identical=bool(np.array_equal(a.iloc[:, :12].values, b.iloc[:, :12].values)), currents_rel_diff=q(np.abs(A - B)[m] / np.abs(B)[m]), max_abs_diff_A=float(np.abs(A - B).max()))
    out["corners_identical"] = bool(pd.read_csv(os.path.join(R, "data", "corners_official.csv")).equals(pd.read_csv(os.path.join(DATA, "corners_official.csv"))))
    a = pd.read_csv(os.path.join(R, "data", "gate_spice_official.csv")); b = pd.read_csv(os.path.join(DATA, "gate_spice_official.csv"))
    out["gate_spice"] = dict(rows_equal_keys=bool((a[["kind", "a", "b", "corner"]].values == b[["kind", "a", "b", "corner"]].values).all()), i_vdd_rel_diff=q(np.abs(a.i_vdd / b.i_vdd - 1)), i_tot_rel_diff=q(np.abs(a.i_tot_vdd / b.i_tot_vdd - 1)),
                           all_ok=bool(a.ok.all()), attempts_new=a.attempt.value_counts().to_dict(), attempts_stored=b.attempt.value_counts().to_dict(), solver_msg_new=int(a.solver_failed_msg.sum()), solver_msg_stored=int(b.solver_failed_msg.sum()))
    ca = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "work", "circuit_spice", "corner_*.csv")))]); cb = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "work", "circuit_spice", "corner_*.csv")))])
    out["circuit_spice"] = dict(n_new=len(ca), all_ok=bool(ca.ok.all()), i_tot_rel_diff=q(np.abs(ca.i_tot.values / cb.i_tot.values - 1)), i_vdd_rel_diff=q(np.abs(ca.i_vdd.values / cb.i_vdd.values - 1)), attempts_new=ca.attempt.value_counts().to_dict(), attempts_stored=cb.attempt.value_counts().to_dict())
    for nm, f in (("test_summary", "tables/test_summary.csv"),):
        n = pd.read_csv(os.path.join(R, "out", f)); s = pd.read_csv(os.path.join(OUT, f)); out[nm] = dict(rerun=n[["case", "model", "med_err_pct", "p95_err_pct"]].round(3).to_dict("records"), stored=s[["case", "model", "med_err_pct", "p95_err_pct"]].round(3).to_dict("records"))
    out["gate_metrics"] = dict(rerun=json.load(open(os.path.join(R, "out", "gate", "gate_overall_metrics.json"))), stored=json.load(open(os.path.join(OUT, "gate", "gate_overall_metrics.json"))))
    cn = json.load(open(os.path.join(R, "out", "circuit", "circuit_overall_metrics.json"))); cs = json.load(open(os.path.join(OUT, "circuit", "circuit_overall_metrics.json")))
    out["circuit_metrics"] = {k: dict(rerun=cn[k]["mean_accuracy_pct"], stored=cs[k]["mean_accuracy_pct"]) for k in ("T", "S", "S_load_corrected")}
    os.makedirs(os.path.join(OUT, "audit"), exist_ok=True); json.dump(out, open(os.path.join(OUT, "audit", "rerun_comparison.json"), "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str)[:6000])
