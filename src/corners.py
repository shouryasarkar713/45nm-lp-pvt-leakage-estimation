"""200 PVT corners (Parts B and C): VDD, T, one NMOS process draw, one PMOS process draw (all U(+-10%) per assignment)."""
import numpy as np, pandas as pd, os
from config import *
def make_corners(n=N_CORNERS, seed=SEEDS["corners"]):
    rng = np.random.default_rng(seed)
    d = {"corner": np.arange(n), "vdd": rng.uniform(VDD_LO, VDD_HI, n), "temp": rng.uniform(T_LO, T_HI, n)}
    for dev in ("nmos", "pmos"):
        for k in PROC:
            nom = NOMINAL[dev][k]; a, c = 0.9 * nom, 1.1 * nom
            d[f"{dev}_{k}"] = rng.uniform(min(a, c), max(a, c), n)
    return pd.DataFrame(d)
def corner_procs(r):
    return ({k: r[f"nmos_{k}"] for k in PROC}, {k: r[f"pmos_{k}"] for k in PROC})
def load_corners(name="official"):
    p = os.path.join(DATA, f"corners_{name}.csv")
    return pd.read_csv(p)
if __name__ == "__main__":
    make_corners().to_csv(os.path.join(DATA, "corners_official.csv"), index=False, float_format="%.10g")
    make_corners(200, 777).to_csv(os.path.join(DATA, "corners_dev.csv"), index=False, float_format="%.10g")
