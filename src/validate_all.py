"""Automated final validation.  Exit code 1 and no 'ALL MANDATORY CHECKS PASSED' unless every mandatory check passes.
Principal metrics are RE-COMPUTED here from the raw CSV files and compared with the reported tables."""
import os, sys, json, glob, itertools, hashlib, inspect, time
import numpy as np, pandas as pd
from config import *
res = []
def check(name, ok, detail=""):
    res.append((name, bool(ok), detail)); print(("PASS " if ok else "FAIL ") + name + ("  -- " + str(detail) if detail else ""), flush=True)
COLS = ["vdd","temp","w","vd","vs","vg","vb","vth0","toxe","ndep","u0","xj","id","ig","is","ib"]
tol = lambda a, b, rel=1e-6: abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)

# ---------------------------------------------------------------- A. transistor datasets
for case in CASES:
    f = os.path.join(DATA, "base", f"{case}.csv"); ok = os.path.exists(f); check(f"A1 dataset exists {case}", ok)
    if not ok: continue
    d = pd.read_csv(f); dev, g, b = CASE_DEF[case]
    check(f"A2 {case}: 10,000 rows and exact columns", len(d) == 10000 and list(d.columns) == COLS)
    check(f"A3 {case}: no NaN / no duplicate rows", not d.isna().any().any() and not d.duplicated().any())
    check(f"A4 {case}: VDD, T, W, Vd, Vs ranges", d.vdd.between(.9, 1.1).all() and d.temp.between(-40, 125).all() and d.w.between(W_LO * .9999, W_HI * 1.0001).all()
          and (d.vd >= 0).all() and (d.vd <= d.vdd).all() and (d.vs >= 0).all() and (d.vs <= d.vdd).all())
    vg = d.vdd if g == "VDD" else 0 * d.vdd; vb = d.vdd if b == "VDD" else 0 * d.vdd
    check(f"A5 {case}: gate/body voltages fixed by case", np.allclose(d.vg, vg) and np.allclose(d.vb, vb))
    pr = []
    for k in PROC:
        nom = NOMINAL[dev][k]; lo, hi = sorted((.9 * nom, 1.1 * nom)); pr.append(d[k].between(lo * (1 - 1e-9), hi * (1 + 1e-9)).all())
    check(f"A6 {case}: process parameters within nominal +-10 %", all(pr))
    I = d[["id", "ig", "is", "ib"]].values; check(f"A7 {case}: KCL |sum of 4 terminal currents| < 1e-12 A", np.abs(I.sum(1)).max() < 1e-12, f"max {np.abs(I.sum(1)).max():.2e}")
    check(f"A8 {case}: uniform-draw sanity (means within 5 % of range centre)", abs(d.temp.mean() - 42.5) < 3 and abs(d.vdd.mean() - 1.0) < .01)
for fn in ("kirchhoff_check.csv", "trend_ioff_vs_temperature.png", "trend_ioff_vs_vth0.png", "trend_ig_vs_toxe.png", "kirchhoff_histograms.png"):
    check(f"A9 data-check artifact {fn}", os.path.exists(os.path.join(OUT, "data_check", fn)))

# ---------------------------------------------------------------- B. models
TD = os.path.join(OUT, "tables"); MD = os.path.join(OUT, "models")
from ml_data import load_base, load_train, load_val_test, features
from split import split_idx
tr_i, va_i, te_i = split_idx()
check("B1 split 7000/1500/1500, disjoint, covers 10,000", len(tr_i) == 7000 and len(va_i) == 1500 and len(te_i) == 1500 and len(set(tr_i) | set(va_i) | set(te_i)) == 10000)
for case in CASES:
    S = pd.read_csv(os.path.join(TD, f"val_summary_{case}.csv")); ext = S[S.data == "extended"]
    fams = set(ext.model); check(f"B2 {case}: four required families + StructMLP compared on validation", {"Ridge", "RandomForest", "HistGradBoost", "MLP"} <= fams and "StructMLP" in fams)
    sel = json.load(open(os.path.join(TD, f"selection_{case}.json")))
    check(f"B3 {case}: selected model = argmin validation median error", sel["chosen"] == ext.loc[ext.med_err_pct.idxmin()].model)
    check(f"B4 {case}: chosen model file exists", os.path.exists(os.path.join(MD, f"model_{case}.joblib")))
    # leakage test: no val/test row (nor its twin) in the training table
    va, te = load_val_test(case); hz = np.load(os.path.join(OUT, "tables", f"row_hashes_{case}.npz"))
    from make_train_manifest import hashes
    check(f"B5a {case}: stored validation/test row hashes match the data files", set(hashes(va)) == set(hz["val"]) and set(hashes(te)) == set(hz["test"]))
    check(f"B5 {case}: zero val/test rows in the training table (hash manifest)", not (set(hz["train"]) & (set(hz["val"]) | set(hz["test"]))), f"{len(hz['train'])} training rows")
    check(f"B6 {case}: test evaluated after selection", os.path.getmtime(os.path.join(TD, "test_summary.csv")) >= os.path.getmtime(os.path.join(TD, f"selection_{case}.json")))
    check(f"B7 {case}: pred-vs-SPICE plot", os.path.exists(os.path.join(OUT, "plots", f"pred_vs_spice_{case}.png")))
    from train_models import load_chosen
    m, kind = load_chosen(case); p = m.predict(te.iloc[:50]) if kind == "StructMLP" else m.predict(features(te.iloc[:50], case))
    check(f"B8 {case}: saved model loads and predicts", np.isfinite(p.values).all())
T = pd.read_csv(os.path.join(TD, "test_summary.csv")); check("B9 test summary has four cases", len(T) == 4)

# ---------------------------------------------------------------- C. gates
G = pd.read_csv(os.path.join(DATA, "gate_spice_official.csv")); P = pd.read_csv(os.path.join(OUT, "gate", "gate_results_per_point.csv"))
check("C1 3,600 gate reference records, all accepted", len(G) == 3600 and G.ok.all())
combos = G.groupby(["kind", "a", "b"]).corner.nunique()
check("C2 18 gate/input cases x 200 corners", len(combos) == 18 and (combos == 200).all())
corn = pd.read_csv(os.path.join(DATA, "corners_official.csv"))
check("C3 200 official corners; gate records use them", len(corn) == 200 and set(G.corner) == set(range(200)))
check("C4 3,600 ML gate predictions; output + internal-node voltages recorded (SPICE and ML)", len(P) == 3600 and P.ml_T.notna().all() and P.v_out.notna().all()
      and P[P.kind == "NAND2"].v_x.notna().all() and P[P.kind == "NOR2"].v_y.notna().all() and P[P.kind == "AND2"].v_G1_x.notna().all() and P[P.kind == "AND2"].v_m.notna().all()
      and P[P.kind == "OR2"].v_G1_y.notna().all() and P[P.kind == "OR2"].v_m.notna().all() and P.loc[P.kind != "INV", "ml_stack_node"].notna().all())
for q, (ref, ml) in {"T": ("i_tot_vdd", "ml_T"), "S": ("i_vdd", "ml_S")}.items():
    tab = pd.read_csv(os.path.join(OUT, "gate", f"gate_results_table_{q}.csv")); e = np.abs(P[ml] - P[ref]) / P[ref]
    check(f"C5 gate table {q}: 18 rows", len(tab) == 18)
    ok = True
    for r in tab.itertuples():
        s = P[(P.kind == r.gate) & (P.a == r.a) & ((P.b == (-1 if r.b == "" or pd.isna(r.b) else int(r.b))))]
        ee = np.abs(s[ml] - s[ref]) / s[ref]; ok &= tol(100 * np.median(ee), r.median_err_pct, 1e-4) and tol(100 * np.quantile(ee, .95), r.p95_err_pct, 1e-4)
    check(f"C6 gate table {q} independently recomputed from per-point CSV", ok)
    check(f"C7 gate overall {q} mean accuracy recomputed", tol(100 * (1 - e.mean()), json.load(open(os.path.join(OUT, "gate", "gate_overall_metrics.json")))[q]["mean_accuracy_pct"], 1e-6))
check("C8 gate parity plots", os.path.exists(os.path.join(OUT, "gate", "gate_parity_T.png")) and os.path.exists(os.path.join(OUT, "gate", "gate_parity_S.png")))
no_ml = "ml_" not in " ".join(c for c in G.columns); check("C9 gate SPICE reference file has no ML-derived column (acceptance independent of ML)", no_ml)

# ---------------------------------------------------------------- D. circuit
from circuit import GATES, logic, OUTPUT
check("D1 truth table = 8-bit parity", all(logic(b)[OUTPUT] == sum(b) % 2 for b in itertools.product([0, 1], repeat=8)))
check("D2 21 gates", len(GATES) == 21)
Cc = pd.read_csv(os.path.join(OUT, "circuit", "circuit_results_per_point.csv.gz"))
check("D3 51,200 circuit reference records (256 x 200)", len(Cc) == 51200 and Cc.groupby("corner").vec.nunique().eq(256).all() and Cc.corner.nunique() == 200)
check("D4 51,200 hierarchical predictions", Cc.ml_T.notna().all() and Cc.ml_S.notna().all())
check("D5 all circuit SPICE references accepted by ML-independent criteria", (Cc.ok == 1).all())
import circuit_eval
src = inspect.getsource(circuit_eval.hierarchical_predict)
check("D6 prediction function never references circuit-SPICE data", "circuit_spice" not in src and "load_circuit_spice" not in src and "i_vdd" not in src and "i_tot" not in src)
# dynamic test: the prediction depends only on (bits, corner, Part-B gate table)
gt = {(r.kind, int(r.a), int(r.b) if r.b >= 0 else int(r.a), int(r.corner)): r.ml_T for r in P.itertuples()}
gt.update({(r.kind, int(r.a), 0, int(r.corner)): r.ml_T for r in P.itertuples() if r.kind == "INV"})
sub = circuit_eval.hierarchical_predict(gt, [0, 7, 123]); mm = Cc.merge(sub, on=["corner", "vec"], suffixes=("", "_re"))
check("D7 hierarchical prediction re-computed from Part-B table only (3 corners x 256 vectors)", len(mm) == 768 and np.allclose(mm.ml_T, mm.ml, rtol=1e-9))
check("D8 circuit table & parity plot", os.path.exists(os.path.join(OUT, "circuit", "circuit_results_table_T.csv")) and os.path.exists(os.path.join(OUT, "circuit", "circuit_parity_T.png")))
e = np.abs(Cc.ml_T - Cc.i_tot) / Cc.i_tot; J = json.load(open(os.path.join(OUT, "circuit", "circuit_overall_metrics.json")))
check("D9 circuit overall metrics recomputed", tol(100 * (1 - e.mean()), J["T"]["mean_accuracy_pct"], 1e-6) and tol(100 * np.median(e), J["T"]["median_err_pct"], 1e-6))

# ---------------------------------------------------------------- F. solver audit / definitions / statistics
sw = pd.read_csv(os.path.join(OUT, "audit", "solver_warning_audit.csv")); Gd = pd.read_csv(os.path.join(DATA, "gate_spice_official.csv"))
check("F1 solver-warning audit covers every gate reference with solver_failed_msg = 1", len(sw) == int((Gd.solver_failed_msg == 1).sum()) and len(sw) > 0, f"{len(sw)} records")
check("F2 every warning-bearing reference is classified valid (no invalid / questionable)", set(sw.classification) <= {"strict_converged", "strict_with_warning", "relaxed_documented"}, str(sw.classification.value_counts().to_dict()))
check("F3 audited re-solves agree within 1 % and KCL < 1e-3", (sw.resolve_max_rel_diff <= 0.01).all() and (sw.kcl_rel <= 1e-3).all())
dv = json.load(open(os.path.join(OUT, "audit", "definition_verification.json")))
check("F4 T = static power / VDD verified from raw source currents (gate and circuit)", dv["gate_T_equals_static_power_over_VDD_max_rel_diff"] < 1e-12 and dv["circuit_T_equals_static_power_over_VDD_max_rel_diff"] < 1e-12)
check("F5 load-corrected S composition verified on SPICE data only (mean error < 0.1 %)", dv["load_correction_SPICE_only_mean_abs_err_pct"] < 0.1)
ex = pd.read_csv(os.path.join(OUT, "tables", "test_terminals_extended.csv"))
check("F6 extended test statistics (signed-scale R2, log R2, KCL) present for all cases", {"r2_signed", "r2_log10_floor1e30", "r2_log10_floor1e15"} <= set(ex.columns) and ex.case.nunique() == 4)
check("F7 quantity labelled in every result table", all("quantity" in pd.read_csv(os.path.join(OUT, d, f)).columns for d, f in (("gate", "gate_results_table_T.csv"), ("gate", "gate_results_table_S.csv"), ("circuit", "circuit_results_table_T.csv"), ("circuit", "circuit_results_table_S.csv"))))
rp = os.path.join(OUT, "audit", "report_check.json")
check("F8 rendered report verified against result files (verify_report.py)", os.path.exists(rp) and json.load(open(rp))["ok"])
check("F9 report PDF and DOCX present", all(os.path.exists(os.path.join(ROOT, "report", f)) for f in ("Project1_Report_45nmLP_1V_ParityChecker.pdf", "Project1_Report_45nmLP_1V_ParityChecker.docx")))


# ---------------------------------------------------------------- G. solver audit detail / DC branch / per-vector tables
swc = pd.read_csv(os.path.join(OUT, "audit", "solver_warning_summary.csv"))
check("G1 solver-warning counts consistent (audit rows = flagged records = summary total)", len(sw) == int((Gd.solver_failed_msg == 1).sum()) == int(swc.n_records.sum()))
check("G2 classification totals recomputed from the audit CSV match the summary", sw.classification.value_counts().to_dict() == dict(zip(swc.classification, swc.n_records)))
check("G3 valid alternative solutions per record: min >= 10, max <= 15 (recomputed from CSV)", sw.n_valid_resolves.min() >= 10 and sw.n_valid_resolves.max() <= 15, f"{sw.n_valid_resolves.min()}-{sw.n_valid_resolves.max()} median {sw.n_valid_resolves.median():g}")
rd = pd.read_csv(os.path.join(OUT, "audit", "solver_repeatability_detail.csv"))
check("G4 repeatability detail: two comparisons per warning record, all pass the stated tolerance", len(rd) == 2 * len(sw) and rd.passes_tolerance.all() and set(rd.comparison) == {"run1_vs_run2", "run1_vs_stored_record"})
check("G4b repeatability compares the full set of quantities (>= 50 per run-vs-run comparison)", rd[rd.comparison == "run1_vs_run2"].n_quantities.min() >= 50)
dcd = pd.read_csv(os.path.join(OUT, "audit", "dc_branch_probe_detail.csv")); dcp = pd.read_csv(os.path.join(OUT, "audit", "dc_branch_probe.csv")); dcs = pd.read_csv(os.path.join(OUT, "audit", "dc_branch_summary.csv"))
check("G5 DC-branch probe: classes recomputed from detail match the summary and sum to all attempts", dcd.classification.value_counts().to_dict() == dict(zip(dcs.classification, dcs.n_solutions_attempted)) and len(dcd) == int(dcs.n_solutions_attempted.sum()))
check("G5b every solution classified valid has finite output, correct logic, positive power, KCL < 1e-3 and repeatable", dcd[dcd.classification == "valid_physically_consistent"].eval("logic_ok == 1 and power_positive == 1 and kcl_rel <= 0.001 and repeat_max_rel_diff <= 1e-9").all())
check("G5c probe-level counts: n_valid > 0 + no_valid_solution_found = number of probes", int((dcp.n_valid > 0).sum()) + int((dcp.probe_result == "no_valid_solution_found").sum()) == len(dcp), f"{len(dcp)} probes, {int((dcp.n_valid == 0).sum())} without valid solution")
Dpt = pd.read_csv(os.path.join(OUT, "circuit", "circuit_results_per_point.csv.gz")); CJ = json.load(open(os.path.join(OUT, "circuit", "circuit_overall_metrics.json")))
for q, fn, ref, ml, gk in (("T", "circuit_results_per_vector_T.csv", "i_tot", "ml_T", "T"), ("S", "circuit_results_per_vector_S.csv", "i_vdd", "ml_S", "S"), ("S_loadcorrected", "circuit_results_per_vector_S_loadcorrected.csv", "i_vdd", "ml_S_loadcorr", "S_load_corrected")):
    pv = pd.read_csv(os.path.join(OUT, "circuit", fn), dtype={"input_bits": str}); req = ["spice_median", "ml_median", "mean_spice", "mean_ml", "bias_pct", "median_error_pct", "p95_error_pct", "mean_mape_pct", "mean_accuracy_pct"]
    ok = len(pv) == 256 and sorted(pv.vector) == list(range(256)) and pv.vector.is_unique and (pv.pvt_count == 200).all()
    ok &= all(pv.input_bits == [format(v, "08b") for v in pv.vector]) and all(pv.parity == [bin(v).count("1") % 2 for v in pv.vector]) and all(pv.hamming_weight == [bin(v).count("1") for v in pv.vector])
    check(f"G6 per-vector {q}: 256 unique vectors 0..255, 200 PVT each, correct bits/parity/Hamming weight", ok)
    check(f"G6b per-vector {q}: no missing/non-finite required metrics", np.isfinite(pv[req].values).all())
    chk = Dpt.groupby("vec").apply(lambda s_: 100 * np.mean(np.abs((s_[ml] - s_[ref]) / s_[ref])) if True else 0, include_groups=False).reindex(range(256)).values
    check(f"G7 per-vector {q}: recomputed from the 51,200 records, and its mean reproduces the global accuracy", np.allclose(chk, pv.mean_mape_pct.values, rtol=1e-9) and abs((100 - pv.mean_mape_pct.mean()) - CJ[gk]["mean_accuracy_pct"]) < 1e-9)

import re as _re
_pat = _re.compile(r"(?i)(previous (implementation|solution|team|version)|earlier (agent|version|implementation|revision)|\bGLM\b|\bagent\b|corrected version|final pass|\bkaggle\b)")
_hits = [f for f in glob.glob(os.path.join(ROOT, "src", "*.py")) + [os.path.join(ROOT, "README.md"), os.path.join(ROOT, "run_all.sh")]
         if os.path.basename(f) not in ("validate_all.py", "verify_report.py") and _pat.search(open(f).read())]
check("F10 no development-history or agent wording in source, README and documentation", not _hits, str(_hits))
check("F11 no local machine paths in documentation and environment record", not any("/home/" in open(f).read() for f in [os.path.join(ROOT, "README.md"), os.path.join(OUT, "environment.json")]))

# ---------------------------------------------------------------- E. netlists / reproducibility
for f in [f"example_transistor_{c}.cir" for c in CASES] + [f"example_gate_{k}.cir" for k in ["INV", "NAND2", "NOR2", "AND2", "OR2"]] + ["example_parity_checker.cir", "45nm_LP.pm"]:
    p = os.path.join(ROOT, "netlists", f); ok = os.path.exists(p) and (f.endswith(".pm") or "gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9" in open(p).read())
    check(f"E1 netlist {f} (+ mandated .options)", ok)
for c in CASES: check(f"E2 {c} netlist has one source per terminal", all(open(os.path.join(ROOT, "netlists", f"example_transistor_{c}.cir")).read().count(x) >= 1 for x in ("Vd ", "Vg ", "Vs ", "Vb ")))
# E6: an example netlist (altermod style) re-simulated with ngspice reproduces the generator's inline-model-card result
import subprocess, re
from spice import run_netlist, parse_prints, gate_devices, build_netlist, LADDER
from gen_gate import sim_ref
from corners import load_corners, corner_procs
def _ex():
    txt = open(os.path.join(ROOT, "netlists", "example_gate_AND2.cir")).read(); env = dict(os.environ); ld = os.environ.get("NGSPICE_LD_LIBRARY_PATH")
    if ld: env["LD_LIBRARY_PATH"] = ld + ":" + env.get("LD_LIBRARY_PATH", "")
    from spice import ngspice_bin
    out = subprocess.run([ngspice_bin(), "-b", "example_gate_AND2.cir"], capture_output=True, text=True, env=env, cwd=os.path.join(ROOT, "netlists")).stdout
    p = parse_prints(out); c = load_corners("official").iloc[0]; nm, pm = corner_procs(c)
    q = sim_ref("AND2", (1, 1), c.vdd, c.temp, nm, pm, 0)[0]
    return abs(p["i(vdd)"] / q["i(vdd)"] - 1) < 1e-6 and abs(p["v(out)"] - q["v(out)"]) < 1e-9
try: ok = _ex()
except Exception as e: ok = False
check("E6 example AND2 netlist (altermod) reproduces the generator's inline model-card result in ngspice", ok)
orc = json.load(open(os.path.join(OUT, "audit", "oracle_reconstruction.json")))["official"]
check("E7 supply reconstruction validated with SPICE device currents (median rel. error < 1e-4)", orc["S"]["median"] < 1e-4 and orc["T"]["median"] < 1e-4)
check("E3 model card present and hash recorded", os.path.exists(MODEL_CARD) and os.path.exists(os.path.join(OUT, "environment.json")))
_internal = [os.path.relpath(os.path.join(dp, f), ROOT) for dp, _, fn in os.walk(ROOT) for f in fn if f in ("INSTRUCTOR_QUESTION.md", "REQUIREMENTS_MATRIX.md") and "provenance" not in os.path.relpath(dp, ROOT).split(os.sep)]
check("E4b internal working documents (instructor-question draft, requirements matrix) are outside the submission files", not _internal, str(_internal))
check("E4 requirements.txt and README present", os.path.exists(os.path.join(ROOT, "requirements.txt")) and os.path.exists(os.path.join(ROOT, "README.md")))
mf = os.path.join(ROOT, "MANIFEST_sha256.txt")
if os.path.exists(mf):
    badm = [p for h, p in (l.split("  ", 1) for l in open(mf).read().splitlines()) if not os.path.exists(os.path.join(ROOT, p)) or sha256(os.path.join(ROOT, p)) != h]
    check("E5 package manifest: every entry exists and its sha256 matches", not badm, f"{len(badm)} bad: {badm[:3]}")
else: check("E5 package manifest present (MANIFEST_sha256.txt at the package root)", False, "run make_packages.py")
bad = [n for n, ok, d in res if not ok]
print(f"\n{len(res) - len(bad)}/{len(res)} checks passed")
if bad: print("FAILED:", bad); sys.exit(1)
print("ALL MANDATORY CHECKS PASSED")
