"""Supplementary (Part B record keeping): ML-predicted stack-node voltage obtained by solving KCL at the stack node with the
transistor models (bisection), compared with the SPICE node voltage.  The primary Part-B leakage prediction does NOT use this."""
import numpy as np, pandas as pd
from config import *
from gate_model import gate_structure

def solve_nodes(g, dr, provider, iters=60):
    """g: gate per-point frame (with _idx); dr: device-row frame from predict_gates.  Returns Series ml_node aligned to g.index."""
    out = pd.Series(np.nan, index=g.index)
    for kind, name_t, name_b in (("NAND2", "G_MnA", "G_MnB"), ("AND2", "G1_MnA", "G1_MnB"), ("NOR2", "G_MpA", "G_MpB"), ("OR2", "G1_MpA", "G1_MpB")):
        sel = g[g.kind == kind]
        if len(sel) == 0: continue
        A = dr[(dr.kind == kind) & (dr.dev == name_t)].set_index("rec").loc[sel._idx]; B = dr[(dr.kind == kind) & (dr.dev == name_b)].set_index("rec").loc[sel._idx]
        nmos = kind in ("NAND2", "AND2"); vdd = A.vdd.values
        lo = np.zeros(len(A)); hi = vdd.copy()
        def f(x):
            rows = []
            for D, role in ((A, "A"), (B, "B")):
                d = D.copy()
                if nmos:      # A: source at x ; B: drain at x
                    if role == "A": d["vs"] = x
                    else: d["vd"] = x
                else:         # PMOS stack: A: drain at y ; B: source at y
                    if role == "A": d["vd"] = x
                    else: d["vs"] = x
                rows.append(d)
            cur = []
            for d in rows:
                out_ = np.zeros((len(d), 4)); 
                for case, idx in d.groupby("case").indices.items():
                    sub = d.iloc[idx]; p = provider(case, sub[["vdd", "temp", "w", "vd", "vs", "vg", "vb"] + PROC])
                    out_[idx] = p[["id", "ig", "is", "ib"]].values
                cur.append(out_)
            return (cur[0][:, 2] + cur[1][:, 0]) if nmos else (cur[0][:, 0] + cur[1][:, 2])   # KCL at the stack node
        flo = f(lo)
        for _ in range(iters):
            mid = 0.5 * (lo + hi); fm = f(mid); left = np.sign(fm) == np.sign(flo)
            lo = np.where(left, mid, lo); flo = np.where(left, fm, flo); hi = np.where(left, hi, mid)
        out.loc[sel.index] = 0.5 * (lo + hi)
    return out
