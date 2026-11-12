"""Deliverable 3: Kirchhoff check per case + three trend plots."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import *
OUTD = os.path.join(OUT, "data_check"); os.makedirs(OUTD, exist_ok=True)
rows = []; D = {c: pd.read_csv(os.path.join(DATA, "base", f"{c}.csv")) for c in CASES}
fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
for a, c in zip(ax, CASES):
    d = D[c]; I = d[["id", "ig", "is", "ib"]].values; k = np.abs(I.sum(1)); sc = np.abs(I).max(1)
    rows.append(dict(case=c, rows=len(d), cols_ok=list(d.columns) == ["vdd","temp","w","vd","vs","vg","vb","vth0","toxe","ndep","u0","xj","id","ig","is","ib"],
                     nan=int(d.isna().sum().sum()), duplicates=int(d.duplicated().sum()), max_abs_kcl_A=k.max(), median_abs_kcl_A=np.median(k),
                     p99_abs_kcl_A=np.quantile(k, .99), max_kcl_over_maxI=(k / sc).max(), median_kcl_over_maxI=np.median(k / sc),
                     frac_kcl_below_1e6_of_maxI=(k / sc < 1e-6).mean()))
    a.hist(np.log10(k / sc + 1e-30), bins=40); a.set_title(c); a.set_xlabel("log10(|sum I| / max|I|)")
plt.tight_layout(); plt.savefig(os.path.join(OUTD, "kirchhoff_histograms.png"), dpi=130); plt.close()
pd.DataFrame(rows).to_csv(os.path.join(OUTD, "kirchhoff_check.csv"), index=False)
# trend plots
def scat(ax, x, y, title, xl, yl, c):
    ax.semilogy(x, np.abs(y), ".", ms=2, alpha=.35, label=c); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title); ax.grid(alpha=.3)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for c, a in zip(("NMOS_OFF", "PMOS_OFF"), ax):
    d = D[c]; m = (d.vd - d.vs).abs() > 0.8 * d.vdd     # large |Vds| so the OFF current is the full subthreshold leakage
    scat(a, d.temp[m], d["id"][m], c + " (|Vds|>0.8 VDD)", "Temperature (C)", "|Id| (A)", c)
plt.tight_layout(); plt.savefig(os.path.join(OUTD, "trend_ioff_vs_temperature.png"), dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for c, a in zip(("NMOS_OFF", "PMOS_OFF"), ax):
    d = D[c]; m = (d.vd - d.vs).abs() > 0.8 * d.vdd
    scat(a, d.vth0.abs()[m], d["id"][m], c + " (|Vds|>0.8 VDD)", "|vth0| (V)", "|Id| (A)", c)
plt.tight_layout(); plt.savefig(os.path.join(OUTD, "trend_ioff_vs_vth0.png"), dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for c, a in zip(("NMOS_ON", "PMOS_ON"), ax):
    d = D[c]; scat(a, d.toxe * 1e9, d["ig"], c, "toxe (nm)", "|Ig| (A)", c)
plt.tight_layout(); plt.savefig(os.path.join(OUTD, "trend_ig_vs_toxe.png"), dpi=130); plt.close()
print(pd.DataFrame(rows).to_string())
