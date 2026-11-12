"""Part A.1 - single-transistor SPICE data. Usage: gen_transistor.py base | extra <mode> <n> <seed>"""
import sys, os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from config import *
from spice import sim_transistor

COLS = ["vdd","temp","w","vd","vs","vg","vb","vth0","toxe","ndep","u0","xj","id","ig","is","ib"]

def draw(case, n, rng, mode="uniform"):
    dev, g, b = CASE_DEF[case]
    vdd = rng.uniform(VDD_LO, VDD_HI, n)                  # VDD first
    temp = rng.uniform(T_LO, T_HI, n); w = rng.uniform(W_LO, W_HI, n)
    vd = rng.uniform(0, vdd); vs = rng.uniform(0, vdd)
    P = {}
    for k in PROC:
        nom = NOMINAL[dev][k]; a, c = 0.9 * nom, 1.1 * nom
        P[k] = rng.uniform(min(a, c), max(a, c), n)
    if mode != "uniform":
        vd, vs = special_bias(mode, vdd, n, rng, vd, vs)
    df = pd.DataFrame(dict(vdd=vdd, temp=temp, w=w, vd=vd, vs=vs))
    df["vg"] = vdd if g == "VDD" else 0.0; df["vb"] = vdd if b == "VDD" else 0.0
    for k in PROC: df[k] = P[k]
    return df[COLS[:12]]

def special_bias(mode, vdd, n, rng, vd, vs):
    """Supplementary (training-only) bias sampling.
    gatecorner : (vd,vs) each at an exact rail (0 or VDD) offset by a log-uniform 0..0.3 V excursion
                 (what logic gates actually apply: stack nodes sit within mV of a rail or mid-stack)
    nearzero   : small |vd-vs| (log-uniform 1e-6..VDD) at random common-mode (linear-region / Vds->0 wedge)
    """
    if mode == "gatecorner":
        def rail():
            base = rng.integers(0, 2, n) * vdd
            off = 10 ** rng.uniform(-9, np.log10(0.3), n)
            v = np.where(base == 0, off, vdd - off)
            exact = rng.random(n) < 0.3
            return np.where(exact, base, v)
        vd = np.where(rng.random(n) < 0.7, rail(), vd); vs = np.where(rng.random(n) < 0.7, rail(), vs)
    elif mode == "nearzero":
        cm = rng.uniform(0, vdd); d = 10 ** rng.uniform(-6, 0, n) * rng.choice([-1, 1], n)
        vd = np.clip(cm + d / 2, 0, vdd); vs = np.clip(cm - d / 2, 0, vdd)
        same = rng.random(n) < 0.1; vs = np.where(same, vd, vs)
    else: raise ValueError(mode)
    return vd, vs

def simulate(case, df, procs=16):
    dev = CASE_DEF[case][0]
    args = [(case, r.vdd, r.temp, r.w, r.vd, r.vs, {k: getattr(r, k) for k in PROC}) for r in df.itertuples()]
    with Pool(procs) as p: res = p.map(sim_transistor, args, chunksize=50)
    cur = np.array(res); out = df.copy()
    for i, c in enumerate(["id", "ig", "is", "ib"]): out[c] = cur[:, i]
    return out[COLS]

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "base":
        os.makedirs(os.path.join(DATA, "base"), exist_ok=True)
        for case, seed in zip(CASES, SEEDS["spec"]):
            t = time.time(); rng = np.random.default_rng(seed)
            df = simulate(case, draw(case, N_SPEC, rng))
            print(case, len(df), "NaN rows:", int(df.isna().any(axis=1).sum()), f"{time.time()-t:.0f}s", flush=True)
            df.to_csv(os.path.join(DATA, "base", f"{case}.csv"), index=False, float_format="%.10g")
    else:
        sub, n, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        os.makedirs(os.path.join(DATA, "extra"), exist_ok=True)
        for ci, case in enumerate(CASES):
            rng = np.random.default_rng(seed * 10 + ci)
            df = simulate(case, draw(case, n, rng, sub))
            df.to_csv(os.path.join(DATA, "extra", f"{case}_{sub}_{seed}.csv"), index=False, float_format="%.10g")
            print(case, sub, len(df), int(df.isna().any(axis=1).sum()), flush=True)
