"""Repeatability of the warning-bearing gate references: every record is re-simulated twice at its accepted solver level and compared with
(a) each other and (b) the stored record, over ALL available quantities: supply/input/ground source currents, every recorded node voltage,
the terminal voltages of every transistor (derived from the node voltages), the SPICE-native terminal currents of every transistor (single-device
re-simulation at the recorded voltages) and the leakage quantities T and S.
Tolerance: pass if |a-b| <= abs_tol + 1e-9*max(|a|,|b|) with abs_tol = 1e-21 A (currents) / 1e-12 V (voltages).  Relative differences use a floor of 1e-18 A / 1e-9 V."""
import os, numpy as np, pandas as pd
from multiprocessing import Pool
from gen_gate import *
from gate_model import gate_structure

ABS = dict(I=1e-21, V=1e-12); FLOOR = dict(I=1e-18, V=1e-9); REL = 1e-9

def collect(p, devs, nodes, nm, pm, c, ab):
    """name -> (value, kind) for one executed run."""
    q = {}
    for k in ("i(vdd)", "i(vss)", "i(va)", "i(vb)"):
        if k in p: q["src:" + k] = (p[k], "I")
    for n in nodes: q["node:" + n] = (p.get(f"v({n.lower()})", np.nan), "V")
    p = dict(p); p["_in_a"] = c.vdd * ab[0]; p["_in_b"] = c.vdd * ab[1] if len(ab) > 1 else 0
    w, cur = kcl_check(devs, p, nm, pm, c.temp, c.vdd)
    nv = {"vdd": c.vdd, "gss": 0.0, "a": p["_in_a"], "b": p["_in_b"], **{n: p.get(f"v({n.lower()})", np.nan) for n in nodes}}
    for name, typ, dn, gn, sn, bn in devs:
        for t, net in zip("dgsb", (dn, gn, sn, bn)): q[f"vterm:{name}.{t}"] = (nv[net], "V")
        for t, v in zip(("id", "ig", "is", "ib"), cur[name]): q[f"iterm:{name}.{t}"] = (v, "I")
    hi = {"a"} if ab[0] else set()
    if len(ab) > 1 and ab[1]: hi.add("b")
    S = T = 0.0
    for name, typ, dn, gn, sn, bn in devs:
        for net, v in zip((dn, gn, sn, bn), cur[name]):
            if net == "vdd": S += v; T += v
            elif net in hi and net == gn: T += v
    q["leak:S_from_terminal_currents"] = (S, "I"); q["leak:T_from_terminal_currents"] = (T, "I"); q["leak:S_source"] = (-p["i(vdd)"], "I")
    q["leak:T_source"] = (-p["i(vdd)"] + sum(-p.get(f"i(v{k})", 0.0) for k, v in zip("ab", ab) if v), "I")
    return q

def diff(qa, qb):
    keys = [k for k in qa if k in qb and np.isfinite(qa[k][0]) and np.isfinite(qb[k][0])]
    ad = np.array([abs(qa[k][0] - qb[k][0]) for k in keys]); kind = [qa[k][1] for k in keys]
    rel = np.array([ad[i] / max(abs(qa[k][0]), abs(qb[k][0]), FLOOR[kind[i]]) for i, k in enumerate(keys)])
    ok = np.array([ad[i] <= ABS[kind[i]] + REL * max(abs(qa[k][0]), abs(qb[k][0])) for i, k in enumerate(keys)])
    return dict(n_quantities=len(keys), n_exactly_equal=int((ad == 0).sum()), all_exactly_equal=bool((ad == 0).all()), max_abs_diff=float(ad.max()), max_rel_diff=float(rel.max()),
                median_rel_diff=float(np.median(rel)), n_fail_tol=int((~ok).sum()), passes_tolerance=bool(ok.all()), worst_quantity=keys[int(np.argmax(rel))])

def one(args):
    r, = args; kind = r["kind"]; ab = (int(r["a"]), int(r["b"])) if r["b"] >= 0 else (int(r["a"]),)
    C = load_corners("official"); c = C.iloc[int(r["corner"])]; nm, pm = corner_procs(c); runs = []
    for rep in range(2):
        p, devs, inputs, nodes, conv, out = sim_ref(kind, ab, c.vdd, c.temp, nm, pm, int(r["attempt"])); runs.append(collect(p, devs, nodes, nm, pm, c, ab))
    # stored record -> same naming where available
    st = {}; miss = []
    for k, col in (("src:i(vdd)", None),):
        pass
    st["src:i(vdd)"] = (-r["i_vdd"], "I"); st["src:i(vss)"] = (r["i_vss"], "I")
    for n in nodes:
        col = "v_" + n.replace("G_", "")
        if col in r and r[col] == r[col]: st["node:" + n] = (r[col], "V")
        else: miss.append("node:" + n)
    for name, typ, dn, gn, sn, bn in devs:
        for t in ("id", "ig", "is", "ib"):
            col = f"{name}.{t}"
            if col in r and r[col] == r[col]: st[f"iterm:{name}.{t}"] = (r[col], "I")
            else: miss.append(f"iterm:{name}.{t}")
    st["leak:S_source"] = (r["i_vdd"], "I"); st["leak:T_source"] = (r["i_tot_vdd"], "I")
    base = dict(kind=kind, a=int(r["a"]), b=int(r["b"]), corner=int(r["corner"]), accepted_level=int(r["attempt"]))
    d1 = diff(runs[0], runs[1]); d2 = diff(runs[0], st)
    return [dict(**base, comparison="run1_vs_run2", **d1, missing_in_comparison=""), dict(**base, comparison="run1_vs_stored_record", **d2, missing_in_comparison=";".join(miss) or "none (stored file has no source currents of inputs)")]

if __name__ == "__main__":
    g = pd.read_csv(os.path.join(DATA, "gate_spice_official.csv")); f = g[g.solver_failed_msg == 1]
    with Pool(12) as pool: rows = [x for r in pool.map(one, [(r,) for r in f.to_dict("records")]) for x in r]
    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "audit", "solver_repeatability_detail.csv"), index=False)
    for cmp_, s in d.groupby("comparison"):
        print(cmp_, "records", len(s), "all exactly equal:", int(s.all_exactly_equal.sum()), "pass tolerance:", int(s.passes_tolerance.sum()), "max abs %.2e max rel %.2e" % (s.max_abs_diff.max(), s.max_rel_diff.max()), "n_quantities", s.n_quantities.min(), "-", s.n_quantities.max())
