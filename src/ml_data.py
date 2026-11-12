"""Dataset assembly, feature construction and metrics for the transistor models."""
import os, glob
import numpy as np, pandas as pd
from config import *
from split import split_idx

I_FLOOR = 1e-15     # relative-error floor [A] (terminal currents below 1 fA do not influence gate leakage at the 1e-12 A scale)
TERM = ["id", "ig", "is", "ib"]

def load_base(case, twin=True):
    df = pd.read_csv(os.path.join(DATA, "base", f"{case}.csv"))
    return df, (pd.read_csv(os.path.join(DATA, "base", f"{case}_twin.csv")) if twin else None)

def load_train(case, extra=True, modes=("uniform", "gatecorner", "nearzero")):
    """Training table = 70 % of the 10k spec rows (+ optional supplementary SPICE rows).  Each frame carries its twin."""
    df, tw = load_base(case); tr, va, te = split_idx()
    parts = [pd.concat([df.iloc[tr].reset_index(drop=True), tw.iloc[tr].reset_index(drop=True)], axis=1).assign(src="spec")]
    if extra:
        for f in sorted(glob.glob(os.path.join(DATA, "extra", f"{case}_*_?.csv"))):
            if not any(f"_{m}_" in f for m in modes): continue
            tf = f.replace(".csv", "_twin.csv")
            if not os.path.exists(tf): continue
            a = pd.read_csv(f); b = pd.read_csv(tf)
            parts.append(pd.concat([a, b], axis=1).assign(src=os.path.basename(f).split("_")[2]))
    d = pd.concat(parts, ignore_index=True)
    return d.dropna().reset_index(drop=True)

def load_val_test(case):
    df, _ = load_base(case, twin=False); tr, va, te = split_idx()
    return df.iloc[va].reset_index(drop=True), df.iloc[te].reset_index(drop=True)

def features(df, case):
    """Physics-motivated input features (all derived from the assignment's input columns only)."""
    dev = CASE_DEF[case][0]
    f = pd.DataFrame(index=df.index)
    f["vdd"] = df.vdd; f["T"] = df.temp; f["invT"] = 1000.0 / (df.temp + 273.15)
    f["logw"] = np.log10(df.w); f["vd"] = df.vd; f["vs"] = df.vs
    vds = df.vd - df.vs
    f["vds"] = vds; f["asv"] = np.arcsinh(vds / 2.6e-2); f["cm"] = 0.5 * (df.vd + df.vs)
    f["vd_n"] = df.vd / df.vdd; f["vs_n"] = df.vs / df.vdd
    f["vth0"] = df.vth0.abs(); f["ltox"] = np.log10(df.toxe); f["lnd"] = np.log10(df.ndep)
    f["lu0"] = np.log10(df.u0); f["xj"] = df.xj
    f["vth_t"] = df.vth0.abs() * f.invT / 3.3          # vth0/(kT)-like exponent driver
    return f

def lg(x, floor=1e-30): return np.log10(np.maximum(np.abs(x), floor))

def metrics(y, p, floor=I_FLOOR):
    """Per-terminal metrics on signed currents."""
    y = np.asarray(y); p = np.asarray(p)
    rel = np.abs(p - y) / np.maximum(np.abs(y), floor)
    ss_res = np.sum((lg(p, 1e-30) - lg(y, 1e-30)) ** 2); ss_tot = np.sum((lg(y, 1e-30) - lg(y, 1e-30).mean()) ** 2)
    return dict(r2_log=1 - ss_res / ss_tot, med_err_pct=100 * np.median(rel), p95_err_pct=100 * np.quantile(rel, .95),
                mean_err_pct=100 * rel.mean(), mae=np.mean(np.abs(p - y)), rmse=np.sqrt(np.mean((p - y) ** 2)),
                sign_acc_pct=100 * np.mean((np.sign(p) == np.sign(y)) | (np.abs(y) < floor)),
                logerr_med=np.median(np.abs(lg(p) - lg(y))))
