"""Part C.6 - complete 8-bit parity checker SPICE reference: 256 vectors x 200 corners (resumable, one npz per corner).
Acceptance criteria are simulator/physics only (finite solution, Boolean logic levels, T>0, independent node-KCL oracle)."""
import sys, os, time, itertools, re
import numpy as np, pandas as pd
from multiprocessing import Pool
from config import *
from spice import *
from corners import *
from circuit import *

KCL_TOL = 1e-3
DEVS, INTERNAL = devices()
NODES = [g[4] for g in GATES] + INTERNAL
VECS = list(itertools.product([0, 1], repeat=8))

def netlist(bits, r, nm, pm, attempt):
    inputs = {f"p{i}": bits[i] for i in range(8)}
    nl = build_netlist(DEVS, r["vdd"], r["temp"], nm, pm, inputs, NODES, f"parity {bits} attempt {attempt}")
    nl = nl.replace(OPTIONS, LADDER[attempt])
    if attempt == len(LADDER) - 1:
        lg = logic(bits); ns = ".nodeset " + " ".join(f"v({g[4]})={lg[g[4]]*r['vdd']:.4f}" for g in GATES) + "\n"
        nl = nl.replace(".control", ns + ".control", 1)
    return nl

def check(bits, r, nm, pm, p):
    vdd = r["vdd"]; lg = logic(bits)
    if any(k not in p or not np.isfinite(p[k]) for k in ["i(vdd)", "i(vss)"] + [f"v({n.lower()})" for n in NODES]): return "nonconv", np.nan, None
    for g in GATES:
        if abs(p[f"v({g[4]})"] - lg[g[4]] * vdd) > 0.05 * vdd: return "logic", np.nan, None
    v = {"vdd": vdd, "gss": 0.0, **{f"p{i}": vdd * bits[i] for i in range(8)}}
    for n in NODES: v[n] = p[f"v({n.lower()})"]
    dv = [(d[0], d[1], tuple(v[x] for x in (d[2], d[3], d[4], d[5]))) for d in DEVS]
    cur = oracle_batch(dv, nm, pm, r["temp"])
    s = {}; a = {}
    for d in DEVS:
        for node, c in zip((d[2], d[3], d[4], d[5]), cur[d[0]]): s[node] = s.get(node, 0) + c; a[node] = a.get(node, 0) + abs(c)
    T = -p["i(vdd)"] + sum(-p[f"i(vp{i})"] for i in range(8) if bits[i])
    worst = sum(abs(s[n]) for n in s if n not in ("vdd", "gss") and not n.startswith("p")) / max(abs(T), 1e-18)
    if not np.isfinite(sum(sum(c) for c in cur.values())): return "oracle", np.nan, None
    if not worst <= KCL_TOL: return "kcl", worst, cur
    if not (-p["i(vdd)"] + sum(-p[f"i(vp{i})"] for i in range(8) if bits[i]) > 0): return "power_sign", worst, cur
    return "", worst, cur

def run_vec(args):
    vi, r, nm, pm = args; bits = VECS[vi]
    for att in range(len(LADDER)):
        p = parse_prints(run_netlist(netlist(bits, r, nm, pm, att), timeout=600))
        why, worst, cur = check(bits, r, nm, pm, p)
        if not why: break
    rec = dict(corner=int(r["corner"]), vec=vi, ok=int(not why), why=why, attempt=att, kcl_rel=worst,
               i_vdd=-p.get("i(vdd)", np.nan), i_vss=p.get("i(vss)", np.nan))
    for i in range(8): rec[f"i_p{i}"] = -p.get(f"i(vp{i})", np.nan)
    rec["i_tot"] = rec["i_vdd"] + sum(rec[f"i_p{i}"] for i in range(8) if bits[i])
    for n in NODES: rec[f"v_{n}"] = p.get(f"v({n.lower()})", np.nan)
    return rec

if __name__ == "__main__":
    C = load_corners("official"); outdir = os.path.join(WORK, "circuit_spice"); os.makedirs(outdir, exist_ok=True)
    sel = range(len(C)) if len(sys.argv) < 2 else [int(x) for x in sys.argv[1].split(",")]
    nvec = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    t0 = time.time()
    with Pool(16) as pool:
        for ci in sel:
            f = os.path.join(outdir, f"corner_{ci:03d}.csv")
            if os.path.exists(f) and nvec == 256: continue
            r = C.iloc[ci].to_dict(); nm, pm = corner_procs(r)
            recs = pool.map(run_vec, [(vi, r, nm, pm) for vi in range(nvec)], chunksize=2)
            df = pd.DataFrame(recs); df.to_csv(f if nvec == 256 else f + ".test", index=False)
            print(ci, "ok", int(df.ok.sum()), "/", len(df), df[df.ok == 0].why.value_counts().to_dict(), "att", df.attempt.value_counts().to_dict(), f"{time.time()-t0:.0f}s", flush=True)
