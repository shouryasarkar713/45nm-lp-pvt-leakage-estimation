"""Part B.3 - gate-level SPICE reference (plain transistor netlists, no instrumentation).
Usage: gen_gate.py <official|dev>.  Reference acceptance uses ONLY simulator/physics criteria (no ML)."""
import sys, os, time, re
import numpy as np, pandas as pd
from multiprocessing import Pool
from config import *
from spice import *
from corners import *

def expected_logic(kind, ab):
    a = ab[0]; b = ab[1] if len(ab) > 1 else 0
    return {"INV": 1 - a, "NAND2": 1 - (a & b), "NOR2": 1 - (a | b), "AND2": a & b, "OR2": a | b}[kind]

def gate_netlist(kind, ab, vdd, temp, nmos, pmos, attempt):  # attempt indexes spice.LADDER
    a, b = ab[0], (ab[1] if len(ab) > 1 else None)
    devs, internal = gate_devices(kind, "G", "a", "b" if b is not None else None, "out")
    inputs = {"a": a}
    if b is not None: inputs["b"] = b
    nodes = ["out"] + internal
    nl = build_netlist(devs, vdd, temp, nmos, pmos, inputs, nodes, f"{kind} {ab} attempt {attempt}")
    nl = nl.replace(OPTIONS, LADDER[attempt])
    if attempt == len(LADDER) - 1:  # last rung additionally seeds a logic-consistent initial guess
        lv = expected_logic(kind, ab)
        ns = f".nodeset v(out)={lv*vdd:.4f}\n"
        if kind in ("AND2", "OR2"): ns += f".nodeset v(G_m)={(1-lv)*vdd:.4f}\n"
        nl = nl.replace(".control", ns + ".control", 1)
    return nl, devs, inputs, nodes

CONV_FAIL = re.compile(r"(stepping failed|singular matrix|no convergence|timestep too small|doAnalyses.*failed|Error:)", re.I)

def sim_ref(kind, ab, vdd, temp, nmos, pmos, attempt=0):
    nl, devs, inputs, nodes = gate_netlist(kind, ab, vdd, temp, nmos, pmos, attempt)
    out = run_netlist(nl); p = parse_prints(out)
    conv = not CONV_FAIL.search(out.replace("Warning: Dynamic gmin stepping failed", "").replace("Warning: True gmin stepping failed", ""))
    return p, devs, inputs, nodes, conv, out

def oracle_currents(devs, p, nmos, pmos, temp, vdd):
    """SPICE single-device re-simulation at the recorded terminal voltages (SPICE-native terminal currents)."""
    def v(n): return {"vdd": vdd, "gss": 0.0}.get(n, p.get(f"v({n})", p.get("v(g_m)" if n == "m" else "", np.nan)))
    res = {}
    for name, typ, dn, gn, sn, bn in devs:
        volt = []
        for n in (dn, gn, sn, bn):
            if n in ("vdd", "gss"): volt.append({"vdd": vdd, "gss": 0.0}[n])
            elif n in ("a", "b"): volt.append(p[f"_in_{n}"])
            else:
                key = f"v({'g_'+n.split('_')[1] if n.startswith('G_') else n.lower()})"
                volt.append(p[f"v({n.lower()})"])
        w = WP if typ == "pmos" else WN
        res[name] = (volt, sim_device(typ, temp, w, *[volt[0], volt[1], volt[2], volt[3]], nmos if typ == "nmos" else pmos))
    return res

KCL_TOL = 1e-3   # a-priori acceptance threshold on the relative node-KCL residual (independent of any ML)

def kcl_check(devs, p, nmos, pmos, temp, vdd):
    orc = oracle_currents(devs, p, nmos, pmos, temp, vdd); s = {}; ab_ = {}; cur_all = {}
    for d in devs:
        volt, cur = orc[d[0]]
        for node, c in zip((d[2], d[3], d[4], d[5]), cur):
            s[node] = s.get(node, 0) + c; ab_[node] = ab_.get(node, 0) + abs(c)
        cur_all[d[0]] = cur
    # residual metric: sum of |node KCL residuals| relative to the total VDD-level supply current T (the quantity of interest)
    T = -p["i(vdd)"] + sum(-p.get(f"i(v{k})", 0.0) for k in ("a", "b") if p.get(f"_in_{k}", 0) > 0)
    worst = sum(abs(s[n]) for n in s if n not in ("vdd", "gss", "a", "b")) / max(abs(T), 1e-18)
    if any(not np.isfinite(c) for cur in cur_all.values() for c in cur): worst = np.inf
    return worst, cur_all

def run_point(args):
    kind, ab, corner, vdd, temp, nmos, pmos = args
    for attempt in range(len(LADDER)):
        p, devs, inputs, nodes, conv, out = sim_ref(kind, ab, vdd, temp, nmos, pmos, attempt)
        why = ""; worst = np.nan; cur_all = {}
        p["_in_a"] = vdd * ab[0]
        if len(ab) > 1: p["_in_b"] = vdd * ab[1]
        if any((k not in p) or (not np.isfinite(p[k])) for k in ("i(vdd)", "i(vss)", "v(out)")): why = "nonconv"
        elif abs(p["v(out)"] - expected_logic(kind, ab) * vdd) > 0.05 * vdd: why = "logic"
        elif not (p["i(vdd)"] < 0): why = "supply_sign"
        else:
            worst, cur_all = kcl_check(devs, p, nmos, pmos, temp, vdd)
            if not worst <= KCL_TOL: why = "kcl"
        if not why: break
    rec = dict(kind=kind, a=ab[0], b=(ab[1] if len(ab) > 1 else -1), corner=corner, vdd=vdd, temp=temp,
               ok=int(not why), why=why, attempt=attempt, solver_failed_msg=int(not conv), kcl_rel=worst,
               i_vdd=-p.get("i(vdd)", np.nan), i_vss=p.get("i(vss)", np.nan),
               i_in_a=-p.get("i(va)", 0.0), i_in_b=-p.get("i(vb)", 0.0))
    for n in nodes: rec[f"v_{n.replace('G_','')}"] = p.get(f"v({n.lower()})", np.nan)
    rec["i_tot_vdd"] = rec["i_vdd"] + (rec["i_in_a"] if ab[0] == 1 else 0) + (rec["i_in_b"] if (len(ab) > 1 and ab[1] == 1) else 0)
    for dname, cur in cur_all.items():
        for t, c in zip(("id", "ig", "is", "ib"), cur): rec[f"{dname}.{t}"] = c
    return rec

if __name__ == "__main__":
    name = sys.argv[1]; C = load_corners(name)
    jobs = []
    for r in C.to_dict("records"):
        nm, pm = corner_procs(r)
        for kind in ["INV", "NAND2", "NOR2", "AND2", "OR2"]:
            for ab in gate_case(kind): jobs.append((kind, ab, int(r["corner"]), r["vdd"], r["temp"], nm, pm))
    t = time.time()
    with Pool(16) as pool: recs = pool.map(run_point, jobs, chunksize=4)
    df = pd.DataFrame(recs); df.to_csv(os.path.join(DATA, f"gate_spice_{name}.csv"), index=False)
    print(len(df), "records; ok:", int(df.ok.sum()), df[df.ok == 0].why.value_counts().to_dict(),
          "attempts:", df.attempt.value_counts().to_dict(), "kcl_rel max %.2e p99 %.2e" % (df.kcl_rel[df.ok==1].max(), df.kcl_rel[df.ok==1].quantile(.99)), f"{time.time()-t:.0f}s")
