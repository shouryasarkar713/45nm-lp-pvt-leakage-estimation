"""DC-operating-point multi-start probe on cold-corner gates.  Every finite alternative solution is classified:
valid_physically_consistent | kcl_valid_logic_invalid | numerically_questionable | unconverged ; probes without any valid solution: no_valid_solution_found.
This is a LIMITED multi-start experiment (random initial guesses x three relaxed solver levels); it cannot prove global uniqueness."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from gen_gate import *
from gate_model import gate_structure

def solve_once(nl, c, nm, pm, kind, ab, devs, nodes):
    p = parse_prints(run_netlist(nl)); p["_in_a"] = c.vdd * ab[0]; p["_in_b"] = c.vdd * ab[1] if len(ab) > 1 else 0
    fin = "v(out)" in p and all(np.isfinite(p.get(k, np.nan)) for k in ["i(vdd)", "v(out)"])
    return p, fin

def probe(args):
    kind, ab, ci, seed, ref = args
    C = load_corners("official"); c = C.iloc[ci]; nm, pm = corner_procs(c); rng = np.random.default_rng(seed); rows = []
    for trial in range(8):
        ns_vals = None
        for rung in (4, 5, 3):
            nl, devs, inputs, nodes = gate_netlist(kind, ab, c.vdd, c.temp, nm, pm, rung)
            nl = nl.replace(".nodeset", "*.nodeset")
            if ns_vals is None: ns_vals = {n: rng.uniform(0, c.vdd) for n in nodes}
            nl = nl.replace(".control", ".nodeset " + " ".join(f"v({n.lower()})={ns_vals[n]:.4f}" for n in nodes) + "\n.control", 1)
            row = dict(kind=kind, a=ab[0], b=ab[1] if len(ab) > 1 else -1, corner=ci, temp=c.temp, trial=trial, rung=rung)
            p, fin = solve_once(nl, c, nm, pm, kind, ab, devs, nodes)
            if not fin: row.update(classification="unconverged", finite=0); rows.append(row); continue
            p2, fin2 = solve_once(nl, c, nm, pm, kind, ab, devs, nodes)
            T = -p["i(vdd)"] + sum(-p.get(f"i(v{k})", 0.0) for k, v in zip("ab", ab) if v)
            w, _ = kcl_check(devs, p, nm, pm, c.temp, c.vdd)
            logic = abs(p["v(out)"] - expected_logic(kind, ab) * c.vdd) < 0.05 * c.vdd
            rep = max(abs(p[k] / p2[k] - 1) if p2.get(k) else np.inf for k in ["i(vdd)", "v(out)"] if k in p) if fin2 else np.inf
            dv = max(abs(p[f"v({n.lower()})"] - ref[n]) for n in nodes if f"v({n.lower()})" in p and ref.get(n) == ref.get(n))
            row.update(finite=1, i_vdd=-p["i(vdd)"], T=T, v_out=p["v(out)"], logic_ok=int(logic), power_positive=int(T > 0), kcl_rel=w, repeat_max_rel_diff=rep,
                       max_node_dV_vs_reference=dv, T_rel_diff_vs_reference=abs(T / ref["T"] - 1))
            if w <= KCL_TOL and logic and T > 0 and rep <= 1e-9: cls = "valid_physically_consistent"
            elif w <= KCL_TOL and not logic: cls = "kcl_valid_logic_invalid"
            else: cls = "numerically_questionable"
            row["classification"] = cls; row["distinct_from_reference"] = int(cls == "valid_physically_consistent" and (row["T_rel_diff_vs_reference"] > 0.01 or dv > 0.01))
            rows.append(row)
    return rows

if __name__ == "__main__":
    off = pd.read_csv("../data/gate_spice_official.csv")
    cold = off[(off.temp < -20) & (off.kind.isin(["AND2", "OR2", "NOR2", "NAND2"]))].groupby(["kind", "a", "b"]).head(3)
    jobs = []
    for r in cold.itertuples():
        ab = (int(r.a), int(r.b)); devs, internal = gate_structure(r.kind); nodes = ["out"] + [n for n in gate_devices(r.kind, "G", "a", "b", "out")[1]]
        ref = {n: getattr(r, "v_" + n.replace("G_", ""), np.nan) for n in nodes}; ref["T"] = r.i_tot_vdd
        jobs.append((r.kind, ab, int(r.corner), 1, ref))
    with Pool(12) as pool: res = pool.map(probe, jobs)
    D = pd.DataFrame([x for rr in res for x in rr]); D.to_csv("../out/audit/dc_branch_probe_detail.csv", index=False)
    S = []
    for (kind, a, b, ci), s in D.groupby(["kind", "a", "b", "corner"]):
        v = s[s.classification == "valid_physically_consistent"]
        S.append(dict(kind=kind, ab=f"({a}, {b})", corner=ci, T=round(s.temp.iloc[0], 1), n_attempts=len(s), n_valid=len(v), n_unconverged=int((s.classification == "unconverged").sum()),
                      n_kcl_valid_logic_invalid=int((s.classification == "kcl_valid_logic_invalid").sum()), n_questionable=int((s.classification == "numerically_questionable").sum()),
                      I_min=v.i_vdd.min() if len(v) else np.nan, I_max=v.i_vdd.max() if len(v) else np.nan, spread_pct=100 * (v.i_vdd.max() - v.i_vdd.min()) / abs(v.i_vdd.mean()) if len(v) else np.nan,
                      max_node_dV_vs_reference=v.max_node_dV_vs_reference.max() if len(v) else np.nan, n_distinct_branch_candidates=int(v.distinct_from_reference.sum()) if len(v) else 0,
                      probe_result="valid_solutions_found" if len(v) else "no_valid_solution_found"))
    P = pd.DataFrame(S); P.to_csv("../out/audit/dc_branch_probe.csv", index=False)
    cnt = D.classification.value_counts().rename_axis("classification").reset_index(name="n_solutions_attempted"); cnt.to_csv("../out/audit/dc_branch_summary.csv", index=False)
    print(cnt.to_string()); print("probes:", len(P), "with valid solution:", int((P.n_valid > 0).sum()), "no valid:", int((P.n_valid == 0).sum()), "max spread %.3f %%" % P.spread_pct.max(), "max node dV %.2e V" % P.max_node_dV_vs_reference.max(), "distinct candidates:", int(P.n_distinct_branch_candidates.sum()))
    print(P[P.n_valid == 0][["kind", "ab", "corner", "T", "n_attempts", "n_unconverged", "n_questionable"]].to_string())
