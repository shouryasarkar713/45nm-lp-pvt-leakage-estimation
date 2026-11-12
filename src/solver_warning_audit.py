"""Audit of the gate references whose ngspice output contained a solver-failure message (solver_failed_msg = 1).
Predefined, ML-independent classification:
  strict_converged        : accepted at level 0 (assignment options), no failure text
  strict_with_warning     : accepted at level 0, failure text present (a fallback sub-method failed, another converged), AND
                            (reproducible on repeat) AND (all independent re-solves agree within 1 %) AND KCL < 1e-3
  relaxed_documented      : accepted only at a relaxed ladder level, same agreement tests passed
  numerically_questionable: tests pass only partially (agreement 1-5 % or KCL 1e-3..1e-2)
  invalid                 : otherwise"""
import os, re, json, numpy as np, pandas as pd
from multiprocessing import Pool
from gen_gate import *
LOGD = os.path.join(OUT, "audit", "solver_logs"); os.makedirs(LOGD, exist_ok=True)
WARN = re.compile(r"^(Note|Warning|Error).*$", re.M)

def one(args):
    r, = args
    kind = r["kind"]; ab = (int(r["a"]), int(r["b"])) if r["b"] >= 0 else (int(r["a"]),)
    C = load_corners("official"); c = C.iloc[int(r["corner"])]; nm, pm = corner_procs(c)
    res = dict(kind=kind, a=int(r["a"]), b=int(r["b"]), corner=int(r["corner"]), temp=c.temp, vdd=c.vdd, accepted_level=int(r["attempt"]), ref_i_vdd=r["i_vdd"], ref_i_tot=r["i_tot_vdd"])
    # 1. log of the accepted level, run twice (reproducibility)
    outs = []
    for rep in range(2):
        p, devs, inputs, nodes, conv, out = sim_ref(kind, ab, c.vdd, c.temp, nm, pm, int(r["attempt"])); outs.append((p, out))
    name = f"{kind}_{ab[0]}{ab[1] if len(ab)>1 else ''}_c{int(r['corner']):03d}"
    open(os.path.join(LOGD, name + ".log"), "w").write(outs[0][1])
    warns = [w for w in WARN.findall(outs[0][1]) if "libtinfo" not in w and "spinit" not in w]
    lines = [m.group(0) for m in re.finditer(r"^(?:Note|Warning|Error)[^\n]*", outs[0][1], re.M) if "libtinfo" not in m.group(0) and "spinit" not in m.group(0)]
    res["warnings"] = " | ".join(lines)
    res["repeat_identical_idd_vout_only"] = int(all(outs[0][0].get(k) == outs[1][0].get(k) for k in ("i(vdd)", "v(out)")))
    # 2. KCL of the accepted solution
    p0 = outs[0][0]; p0["_in_a"] = c.vdd * ab[0]; p0["_in_b"] = c.vdd * ab[1] if len(ab) > 1 else 0
    res["kcl_rel"] = kcl_check(devs, p0, nm, pm, c.temp, c.vdd)[0]
    res["logic_ok"] = int(abs(p0["v(out)"] - expected_logic(kind, ab) * c.vdd) < 0.05 * c.vdd); res["supply_current_sign_ok"] = int(p0["i(vdd)"] < 0)
    res["power_positive"] = int(-p0["i(vdd)"] + sum(-p0.get(f"i(v{k})", 0) for k, v in zip("ab", ab) if v) > 0)
    # 3. independent re-solves: other ladder levels + random nodesets, compare supply current to the reference
    rng = np.random.default_rng(int(r["corner"])); vals = []
    for lev in (0, 1, 2, 3, 4):
        for trial in range(3):
            nl, devs, inputs, nodes = gate_netlist(kind, ab, c.vdd, c.temp, nm, pm, lev)
            if trial: nl = nl.replace(".control", ".nodeset " + " ".join(f"v({n.lower()})={rng.uniform(0, c.vdd):.4f}" for n in nodes) + "\n.control", 1)
            p = parse_prints(run_netlist(nl)); p["_in_a"] = c.vdd * ab[0]; p["_in_b"] = c.vdd * ab[1] if len(ab) > 1 else 0
            if "v(out)" in p and np.isfinite(p["v(out)"]) and kcl_check(devs, p, nm, pm, c.temp, c.vdd)[0] <= KCL_TOL: vals.append((lev, trial, -p["i(vdd)"]))
    v = np.array([x[2] for x in vals]); res["n_valid_resolves"] = len(v)
    res["resolve_max_rel_diff"] = float(np.max(np.abs(v / res["ref_i_vdd"] - 1))) if len(v) else np.nan
    strict = [x[2] for x in vals if x[0] == 0 and x[1] == 0]
    res["strict_default_converges"] = int(len(strict) > 0)
    res["strict_vs_ref_rel_diff"] = float(abs(strict[0] / res["ref_i_vdd"] - 1)) if strict else np.nan
    # 4. classification
    ok = res["repeat_identical_idd_vout_only"] and res["kcl_rel"] <= KCL_TOL and res["logic_ok"] and res["power_positive"] and res["n_valid_resolves"] >= 5
    d = res["resolve_max_rel_diff"]
    if r["solver_failed_msg"] == 0 and r["attempt"] == 0: cls = "strict_converged"
    elif ok and d <= 0.01: cls = "strict_with_warning" if r["attempt"] == 0 else "relaxed_documented"
    elif res["kcl_rel"] <= 1e-2 and d <= 0.05 and res["logic_ok"]: cls = "numerically_questionable"
    else: cls = "invalid"
    res["classification"] = cls
    return res

if __name__ == "__main__":
    g = pd.read_csv(os.path.join(DATA, "gate_spice_official.csv")); f = g[g.solver_failed_msg == 1]
    with Pool(12) as pool: rows = pool.map(one, [(r,) for r in f.to_dict("records")])
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "audit", "solver_warning_audit.csv"), index=False)
    s = df.classification.value_counts().rename_axis("classification").reset_index(name="n_records"); s.to_csv(os.path.join(OUT, "audit", "solver_warning_summary.csv"), index=False)
    print(s.to_string()); print(df[["kind", "a", "b", "corner", "accepted_level", "kcl_rel", "n_valid_resolves", "resolve_max_rel_diff", "strict_vs_ref_rel_diff", "classification"]].round(5).to_string())
    print(df.warnings.str.split(" | ").explode().value_counts().head(6))
