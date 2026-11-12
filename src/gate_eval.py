"""Gate-level ML leakage prediction (Part B.4) from transistor models + SPICE-recorded node voltages."""
import numpy as np, pandas as pd
from config import *
from gate_model import gate_structure, supply_from_currents
from corners import *

def device_rows(gates_df, corners, kinds=None):
    """One row per (gate record, transistor) with the terminal voltages (from recorded node voltages) and PVT/process params."""
    rows = []
    for kind, g in gates_df.groupby("kind"):
        devs, _ = gate_structure(kind)
        for r in g.to_dict("records"):
            c = corners.iloc[int(r["corner"])]; vdd = r["vdd"]
            nv = {"vdd": vdd, "gss": 0.0, "a": vdd * r["a"], "out": r["v_out"]}
            if r["b"] >= 0: nv["b"] = vdd * r["b"]
            for n in ("G_x", "G_y", "G_m", "G1_x", "G1_y"):
                k = "v_" + n.replace("G_", "")
                if k in r and r[k] == r[k]: nv[n] = r[k]
            for name, typ, dn, gn, sn, bn in devs:
                glev = int(nv[gn] > 0.5 * vdd)
                case = ("NMOS_" if typ == "nmos" else "PMOS_") + (("ON" if glev else "OFF") if typ == "nmos" else ("OFF" if glev else "ON"))
                row = dict(rec=r["_idx"], kind=kind, a=r["a"], b=r["b"], corner=r["corner"], dev=name, typ=typ, case=case,
                           vdd=vdd, temp=r["temp"], w=WP if typ == "pmos" else WN, vd=nv[dn], vs=nv[sn],
                           vg=vdd if glev else 0.0, vb=vdd if bn == "vdd" else 0.0, dn=dn, gn=gn, sn=sn, bn=bn,
                           gate_high=int(gn in ("a", "b") and nv[gn] > 0.5 * vdd))
                for k in PROC: row[k] = c[f"{typ}_{k}"]
                rows.append(row)
    return pd.DataFrame(rows)

def predict_gates(gates_df, corners, provider):
    """provider(case, df)->DataFrame(id,ig,is,ib).  Returns gates_df with ml_S, ml_T and a per-device frame."""
    g = gates_df.reset_index(drop=True).copy(); g["_idx"] = np.arange(len(g))
    dr = device_rows(g, corners)
    for t in ("id", "ig", "is", "ib"): dr["p_" + t] = np.nan
    for case, sub in dr.groupby("case"):
        p = provider(case, sub[["vdd", "temp", "w", "vd", "vs", "vg", "vb"] + PROC])
        for t in ("id", "ig", "is", "ib"): dr.loc[sub.index, "p_" + t] = np.asarray(p[t])
    S = np.zeros(len(g)); T = np.zeros(len(g)); INA = np.zeros(len(g)); INB = np.zeros(len(g))
    for r in dr.itertuples():
        cur = (r.p_id, r.p_ig, r.p_is, r.p_ib)
        for node, i in zip((r.dn, r.gn, r.sn, r.bn), cur):
            if node == "vdd": S[r.rec] += i; T[r.rec] += i
            elif node in ("a", "b") and r.gate_high and node == r.gn:
                T[r.rec] += i
                if node == "a": INA[r.rec] += i
                else: INB[r.rec] += i
    g["ml_S"] = S; g["ml_T"] = T; g["ml_in_a"] = INA; g["ml_in_b"] = INB
    return g, dr

def accuracy(ref, pred):
    e = np.abs(pred - ref) / np.abs(ref)
    return dict(mean_acc=100 * (1 - e.mean()), med_err=100 * np.median(e), p95_err=100 * np.quantile(e, .95), bias=100 * np.median((pred - ref) / ref))
