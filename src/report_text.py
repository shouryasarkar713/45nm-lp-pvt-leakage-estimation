# executed inside make_report.py (shares its namespace)
env = J("environment.json"); gM = J("gate", "gate_overall_metrics.json"); cM = J("circuit", "circuit_overall_metrics.json")
gT = pd.read_csv(os.path.join(OUT, "gate", "gate_results_table_T.csv")); gS = pd.read_csv(os.path.join(OUT, "gate", "gate_results_table_S.csv"))
cT = pd.read_csv(os.path.join(OUT, "circuit", "circuit_results_table_T.csv"))
test = pd.read_csv(os.path.join(OUT, "tables", "test_summary.csv")); testT = pd.read_csv(os.path.join(OUT, "tables", "test_terminals.csv"))
val = pd.concat([pd.read_csv(os.path.join(OUT, "tables", f"val_summary_{c}.csv")) for c in CASES])
rb = pd.read_csv(os.path.join(OUT, "gate", "gate_robustness_T.csv")); wg = pd.read_csv(os.path.join(OUT, "gate", "gate_worst_records.csv")); cw = pd.read_csv(os.path.join(OUT, "circuit", "circuit_worst_records.csv"))
dv = J("audit", "definition_verification.json"); orc = J("audit", "oracle_reconstruction.json")["official"]
sw = pd.read_csv(os.path.join(OUT, "audit", "solver_warning_audit.csv")); rpd = pd.read_csv(os.path.join(OUT, "audit", "solver_repeatability_detail.csv"))
dcd = pd.read_csv(os.path.join(OUT, "audit", "dc_branch_probe_detail.csv")); dcp = pd.read_csv(os.path.join(OUT, "audit", "dc_branch_probe.csv")); lv = pd.read_csv(os.path.join(OUT, "audit", "ladder_vs_strict.csv"))
gstat = pd.read_csv(os.path.join(DATA, "gate_spice_official.csv")); cs = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(WORK, "circuit_spice", "corner_*.csv")))])
pvT = pd.read_csv(os.path.join(OUT, "circuit", "circuit_results_per_vector_T.csv"), dtype={"input_bits": str}); pvS = pd.read_csv(os.path.join(OUT, "circuit", "circuit_results_per_vector_S.csv"), dtype={"input_bits": str})
pvC = pd.read_csv(os.path.join(OUT, "circuit", "circuit_results_per_vector_S_loadcorrected.csv"), dtype={"input_bits": str}); pvs = J("circuit", "circuit_per_vector_summary.json"); an = pvs["association_analysis_T"]
gT_all, cT_all = gM["T"], cM["T"]; cls_ = sw.classification.value_counts(); nv = sw.n_valid_resolves
r12 = rpd[rpd.comparison == "run1_vs_run2"]; rst = rpd[rpd.comparison == "run1_vs_stored_record"]; dcs = dcd.classification.value_counts(); nov = dcp[dcp.n_valid == 0]
rc = J("audit", "rerun_comparison.json") if os.path.exists(os.path.join(OUT, "audit", "rerun_comparison.json")) else None
dev = pd.read_csv(os.path.join(OUT, "dev_ablation.csv")); dev = dev[~dev.model.str.contains("prototype")]
spread_all = rb[rb.group != "gate"].mean_accuracy_pct
pv = lambda x: x if isinstance(x, str) else f"{x:.2f}"

# ------------------------------------------------------------------ title page
for _ in range(5): doc.add_paragraph()
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER; r = t.add_run("ML-Driven PVT-Variation-Aware Leakage Estimation"); r.bold = True; r.font.size = Pt(24); r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER; r = t.add_run("45 nm LP, VDD = 1 V - Single Transistors, CMOS Gates and an 8-Bit Parity Checker"); r.font.size = Pt(14)
doc.add_paragraph()
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER; r = t.add_run("Digital VLSI Design - Course Project 1"); r.font.size = Pt(13)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER; r = t.add_run(f"Technology: PTM 45 nm low-power BSIM4 model card (45nm_LP.pm)   |   Simulator: ngspice-47   |   Test circuit: 8-bit parity checker"); r.font.size = Pt(10.5); r.italic = True
doc.add_page_break()

# ------------------------------------------------------------------ 1 abstract
H("1. Abstract")
Pp(f"Leakage current of CMOS circuits depends strongly on process, voltage and temperature (PVT). This report presents a machine-learning (ML) estimator of static leakage that is based on models of single transistors rather than on complete-circuit SPICE simulation. "
   f"Four single-FET cases (NMOS and PMOS, ON and OFF) of the 45 nm LP BSIM4 model card are characterised by SPICE over random PVT draws; four classical regression families and a physics-structured multilayer perceptron are compared on a held-out validation set, and the selected structured models reach a test median relative error of {test.med_err_pct.min():.2f}-{test.med_err_pct.max():.2f} % per case. "
   "The trained transistor models are then applied to five static CMOS gates (INV, NAND2, NOR2, AND2, OR2; every input combination at 200 PVT corners, 3,600 SPICE references) using the internal-node and output voltages recorded from their respective gate-level SPICE operating-point simulations, as prescribed by the assignment. "
   "The resulting ML-predicted terminal currents are used to estimate gate leakage, which is compared against the corresponding SPICE reference. Finally, the gate models are hierarchically composed to estimate the leakage of an 8-bit parity checker without using the complete-circuit SPICE operating-point data as prediction inputs; "
   "the 51,200 complete-netlist SPICE simulations (256 input vectors x 200 corners) serve as the evaluation reference.")
Pp(f"With the accuracy score Accuracy = 100 (1 - MAPE) and total static source current T as the leakage quantity, the gate-level mean accuracy is {gT_all['mean_accuracy_pct']:.2f} % (median error {gT_all['median_err_pct']:.2f} %, 95th percentile {gT_all['p95_err_pct']:.2f} %) and the hierarchical circuit-level mean accuracy is {cT_all['mean_accuracy_pct']:.2f} % "
   f"(median {cT_all['median_err_pct']:.2f} %, 95th percentile {cT_all['p95_err_pct']:.2f} %). For the supply-rail current alone (S), the plain sum of isolated-gate predictions reaches {cM['S']['mean_accuracy_pct']:.2f} % at circuit level because gate currents drawn through upstream drivers are not part of the isolated-gate quantity; with that load current accounted for, "
   f"the accuracy is {cM['S_load_corrected']['mean_accuracy_pct']:.2f} %. Worst-case errors are {100*float(wg.e.abs().max()):.1f} % for a single gate record and {100*float(cw.err_T.abs().max()):.1f} % for a circuit record. The quantity definitions, the numerical validation of the SPICE references and the limitations are documented in Sections 16-18.")

# ------------------------------------------------------------------ 2 introduction
H("2. Introduction and Project Objectives")
Pp("The project follows a three-part workflow. In Part A, single NMOS and PMOS transistors are simulated in their ON and OFF states under random PVT variation and four terminal-current models (I_D, I_G, I_S, I_B) are trained per case. In Part B, these models are applied to five static CMOS gates for every input combination at 200 PVT corners. "
   "In Part C, the same approach is applied to an 8-bit parity checker: the circuit leakage is the sum of the Part-B gate predictions, compared against SPICE simulation of the complete netlist. "
   "In Part B, gate-level SPICE simulations provide both the prescribed internal-node and output voltages used to construct transistor-model inputs and the reference leakage values for evaluation. In Part C, the complete parity-checker SPICE simulation is used exclusively as the evaluation reference; the hierarchical ML prediction uses the Part-B gate predictions.")
T(pd.DataFrame([["1", "Four 10,000-row transistor datasets", "Section 6", "data/base/*.csv"], ["2", "Example transistor, gate and circuit netlists", "Sections 5, 9, 12", "netlists/"], ["3", "Kirchhoff check and three trend plots", "Section 6", "out/data_check/"],
   ["4", "Model comparison, 4 families x 4 cases", "Section 8", "out/tables/val_summary_*.csv"], ["5", "Chosen models, test metrics, predicted-vs-SPICE plots", "Section 8", "out/models/, out/plots/, out/tables/"],
   ["6-7", "Gate results table (18 cases) and parity plot", "Section 11", "out/gate/"], ["8", "Circuit results tables and parity plot (incl. 256 per-vector tables)", "Section 14", "out/circuit/"], ["9", "Code, seeds and run order", "Section 20", "src/, run_all.sh, README.md"], ["10", "This report", "-", "report/"]],
   columns=["No.", "Deliverable", "Report section", "Location in the package"]), size=8, cap="Table 1. Assignment deliverables and where they are documented.")

# ------------------------------------------------------------------ 3 technology
H("3. Technology and Simulation Environment")
T(pd.DataFrame([["Model file", "45nm_LP.pm (PTM low-power 45 nm bulk, BSIM4 level 54), sha256 " + env["model_card_sha256"]], ["Nominal VDD / sweep", "1.0 V; sampled uniformly in 0.9-1.1 V"],
   ["Channel length / widths", "L = 45 nm; transistor datasets W = 45-360 nm (1-8 Lmin); gates and circuit Wn = 90 nm, Wp = 180 nm"], ["Temperature", "uniform, -40 to 125 C"],
   ["Process parameters", "vth0, toxe, ndep, u0, xj: uniform in nominal +-10 %, independent NMOS and PMOS values"], ["Test circuit", "8-bit parity checker"],
   ["Simulator", "ngspice-47; every netlist carries .options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"], ["Software", f"python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, scikit-learn {env['sklearn']}, torch {env['torch']} (training only)"]],
   columns=["Item", "Value"]), size=8.5, cap="Table 2. Assigned configuration and software environment.")
Pp("The supplied file 22n_NMOS_ON.net is a 22 nm conduction-NMOS testbench template (it includes 22nm_MGK.pm). It does not belong to the assigned 45 nm LP configuration; only its structure (one ideal source per terminal, a voltage/temperature sweep) was used as a template, and no 22 nm parameter enters any result.")

# ------------------------------------------------------------------ 4 physics
H("4. Leakage Physics and Mathematical Background")
Pp("In a static CMOS gate, leakage flows through three mechanisms: (i) sub-threshold conduction, I_sub proportional to exp[(V_GS - V_th)/(n kT/q)] (1 - exp(-V_DS/(kT/q))), with drain-induced barrier lowering and a strong dependence on temperature and vth0; "
   "(ii) gate tunnelling, which falls exponentially with oxide thickness toxe and is present in ON devices and across the overlaps; and (iii) junction, body and gate-induced drain leakage. In stacks of series transistors the intermediate node self-biases, which reduces the sub-threshold current (stack effect).")
Pp("Terminal currents are defined as currents flowing into the device (the negative of the ngspice source currents). Charge conservation requires I_D + I_G + I_S + I_B = 0 for every device.")
Pp("Two leakage quantities are evaluated at every gate and circuit operating point. T is the total static current delivered by all voltage sources held at VDD, i.e. the VDD rail plus every input held at logic 1; T = P_static / VDD. S is the current drawn from the VDD rail alone, S = -I(Vdd). "
   "Both are obtained from the transistor terminal currents by summing, over all device terminals connected to the VDD net (and, for T, to inputs at VDD), the current flowing into the device terminal. Their relation is analysed in Section 16.")
Pp("Accuracy metric. For N operating points with SPICE reference I_i and ML prediction I^_i of the same quantity (T or S):", bold=True)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Accuracy_Q = 100 × (1 − MAPE_Q),     MAPE_Q = (1/N) Σ_i | Î_i − I_i | / | I_i |,     Q ∈ { T, S, S_load-corrected }"); r.italic = True; r.font.name = "Cambria"
Pp("This is a regression score, not a classification accuracy. It is reported together with the median and 95th-percentile absolute relative error and the signed bias (the median of (I^ - I)/I), because a mean can hide a heavy error tail. "
   "For single-transistor terminal currents, which can be arbitrarily close to zero, relative errors use the denominator max(|I|, 1 fA); currents below about 1 fA are at the numerical noise level of the SPICE labels (Section 6). No observation is removed or clipped.")

# ------------------------------------------------------------------ 5 single-transistor characterization
H("5. Single-Transistor Characterization")
Pp("Each case uses a single-FET netlist in which every terminal (drain, gate, source, body) has its own ideal voltage source. The gate and body voltages are fixed by the case (NMOS ON: Vg = VDD, Vb = 0; NMOS OFF: Vg = 0, Vb = 0; PMOS ON: Vg = 0, Vb = VDD; PMOS OFF: Vg = VDD, Vb = VDD). "
   "A sample draws VDD first and then T, W, Vd and Vs (uniform on 0-VDD) and the five process parameters (uniform within +-10 % of the model-card value). The process values replace the corresponding model-card parameters, exactly as the ngspice command altermod does; the example netlists use altermod and reproduce the generator to 1e-15 relative. "
   "The DC operating point (.op) is solved at the sampled temperature and the four source currents are recorded (seeds 101-104).")

# ------------------------------------------------------------------ 6 datasets
H("6. Dataset Generation and Statistical Analysis")
Pp("The base datasets contain exactly 10,000 rows per case with the columns vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj, id, ig, is, ib, and are split 70/15/15 (7,000 / 1,500 / 1,500 rows, seed 2024). Validation and test rows come only from these base sets. "
   "Because logic gates operate in regions that uniform sampling covers sparsely (ON devices with |Vds| of 1e-8-1e-6 V, devices at the supply rails), the training set is supplemented with additional SPICE samples generated with the same netlists: 100,000 uniform samples, 100,000 'gate-corner' samples with Vd and Vs within a log-uniform 1 nV-0.3 V excursion of a rail, and 60,000 'near-zero-Vds' samples (|Vd - Vs| log-uniform between 1 uV and 1 V). "
   "Each training row also has a 'zero-Vds twin' (the same sample with Vd set equal to Vs), which gives the Vds = 0 leakage offsets (id0, ig0, is0, ib0); twins exist only for training rows. Each case therefore has 267,000 training rows (7,000 base and 260,000 supplementary). "
   "The base and supplementary SPICE data were generated with ngspice printing 12 significant digits; regeneration with the code of this package prints 15 digits and reproduces the values to about 1e-9 relative.")
kc = pd.read_csv(os.path.join(OUT, "data_check", "kirchhoff_check.csv"))
T(kc.rename(columns={"nan": "missing_values"})[["case", "rows", "missing_values", "duplicates", "max_abs_kcl_A", "median_abs_kcl_A", "p99_abs_kcl_A", "median_kcl_over_maxI"]], size=8, cap="Table 3. Kirchhoff check: absolute value of the sum of the four terminal currents (A) per case.")
Pp("The residual of the current sum stays below 1e-13 A for the ON cases (currents of order 1e-4 A) and below 1e-15 A for the OFF cases. For PMOS OFF the currents are of order 1e-14 A, so a residual of a few 1e-17 A is about 1e-3 of the largest terminal current; "
   "tests with tighter solver tolerances left it unchanged, and the gate-source current of -2.2e-16 A equals one unit in the last place of a 1 V quantity, which identifies the residual as floating-point noise of the source-current evaluation. Labels below about 1 fA are therefore noise-limited.")
IMG(os.path.join(OUT, "data_check", "trend_ioff_vs_temperature.png"), 12.5, "Figure 1. OFF-state |I_D| versus temperature for NMOS and PMOS (samples with |Vds| > 0.8 VDD).")
IMG(os.path.join(OUT, "data_check", "trend_ioff_vs_vth0.png"), 12.5, "Figure 2. OFF-state |I_D| versus |vth0| (samples with |Vds| > 0.8 VDD).")
IMG(os.path.join(OUT, "data_check", "trend_ig_vs_toxe.png"), 12.5, "Figure 3. ON-state |I_G| versus oxide thickness toxe.")
_dec = {}
for _c in ("NMOS_OFF", "PMOS_OFF"):
    _d = pd.read_csv(os.path.join(DATA, "base", f"{_c}.csv")); _m = (_d.vd - _d.vs).abs() > 0.8 * _d.vdd
    _dec[_c] = np.log10(_d[_m & (_d.temp > 110)]["id"].abs().median() / _d[_m & (_d.temp < -30)]["id"].abs().median())
Pp(f"The median OFF current at T > 110 C exceeds that at T < -30 C by {_dec['NMOS_OFF']:.1f} decades (NMOS) and {_dec['PMOS_OFF']:.1f} decades (PMOS) and decreases exponentially with |vth0|, as expected for sub-threshold conduction; the gate current decreases steeply with toxe, as expected for direct tunnelling.")

# ------------------------------------------------------------------ 7 ML methodology
H("7. Machine-Learning Methodology")
Pp("Input features (all derived from the dataset columns): VDD, T, 1000/T, log10 W, Vd, Vs, Vds, asinh(Vds/26 mV), the common mode (Vd + Vs)/2, Vd/VDD, Vs/VDD, |vth0|, log10 toxe, log10 ndep, log10 u0, xj and a vth0/kT-like term. "
   "Currents are never regressed directly: each current is decomposed into log10|I| (regression target) and its sign (a gradient-boosting classifier, used only where both signs occur).")
T(pd.DataFrame([["Ridge", "standardised degree-2 polynomial features, alpha = 1e-2, multi-output"], ["Random Forest", "120 trees, min_samples_leaf 2, max_features 0.7, multi-output"],
   ["Histogram Gradient Boosting", "600 iterations, learning rate 0.08, 63 leaves, L2 1e-3, one regressor per terminal"], ["MLP (scikit-learn)", "(256, 128, 64) ReLU, Adam 2e-3, batch 512, at most 40 epochs, early stopping, multi-output"],
   ["Structured MLP", "5 x 256 SiLU trunk; heads for log10|id0|, log10(|R|/|Vds|), log10|ig|, log10|ib| and sign logits; ensemble of 3 networks (log-domain average); AdamW, one-cycle schedule, 60 epochs, batch 2048, Huber loss (beta 0.05)"]],
   columns=["Model", "Main hyper-parameters"]), size=8, cap="Table 4. Model families compared for each transistor case.")
Pp("Structured model. An ON transistor in a gate sits at |Vds| of 1e-8-1e-6 V and carries about 1e-12 A, while the same device carries 1e-4 A at Vds = 1 V; the drain current also changes sign at Vds = -id0/G. A plain log|I| plus sign model must resolve this zero crossing. "
   "The structured model writes I_D = id0 + sgn(Vds) |Vds| G, where id0 is the leakage offset at Vds = 0 (learned from the twin samples) and G is a smooth conductance-like quantity, learns I_G and I_B directly, and obtains I_S = -I_D - I_G - I_B, so that charge conservation holds by construction. The trained networks are exported as plain numpy weights, so inference needs no deep-learning library.")
Pp("Protocol. The classical families are trained twice: on the 7,000 base training rows ('7k') and on a fixed random subsample of 80,000 rows of the 267,000-row training table ('ext'; limited by compute); the structured MLP uses all 267,000 rows. Model selection (family and number of epochs) uses the validation split only, with the mean over the four terminals of the validation median relative error as criterion. "
   "The test split is evaluated once after selection, and the absence of validation and test rows (and of their twins) from the training table is verified by a row-hash check.")

# ------------------------------------------------------------------ 8 transistor results
H("8. Transistor-Level Results")
v = val[val.data == "extended"].copy(); v7 = val[val.data == "spec7000"].copy(); rows = []
for c in CASES:
    for m in ["Ridge", "RandomForest", "HistGradBoost", "MLP", "StructMLP"]:
        a = v[(v.case == c) & (v.model == m)]; b = v7[(v7.case == c) & (v7.model == m)]
        rows.append([c, m, float(b.r2_log.iloc[0]) if len(b) else np.nan, float(b.med_err_pct.iloc[0]) if len(b) else np.nan, float(b.p95_err_pct.iloc[0]) if len(b) else np.nan, float(a.r2_log.iloc[0]), float(a.med_err_pct.iloc[0]), float(a.p95_err_pct.iloc[0])])
T(pd.DataFrame(rows, columns=["case", "model", "R2 (7k)", "median % (7k)", "p95 % (7k)", "R2 (ext)", "median % (ext)", "p95 % (ext)"]), size=7.5,
  cap="Table 5. Validation comparison of four model families (and the structured MLP) for each case; mean over the four terminals. R2 is on log10|I|. '7k': trained on the 7,000 base rows; 'ext': extended training data.")
_sel = v[v.model == "StructMLP"].set_index("case").med_err_pct; _cl = v[v.model != "StructMLP"].groupby("case").med_err_pct.min()
Pp(f"The structured MLP is selected for every case because it has the lowest validation median error ({_sel.min():.2f}-{_sel.max():.2f} %, against {_cl.min():.1f}-{_cl.max():.1f} % for the best classical family of each case). The classical families are not designed to represent the Vds -> 0 behaviour of ON devices, which is consistent with their larger errors in the ON cases.")
T(test[["case", "model", "r2_log", "med_err_pct", "p95_err_pct", "mean_err_pct", "sign_acc_pct", "logerr_med"]].rename(columns={"r2_log": "R2 log", "med_err_pct": "median %", "p95_err_pct": "p95 %", "mean_err_pct": "mean %", "sign_acc_pct": "sign acc %", "logerr_med": "median |dlog10|"}),
  size=8, cap="Table 6. Selected models on the held-out test split (1,500 rows per case); mean over the four terminals.")
T(testT[["case", "terminal", "r2_log", "med_err_pct", "p95_err_pct", "mean_err_pct", "sign_acc_pct", "mae", "rmse"]], size=7.5, cap="Table 7. Selected models, test split, per terminal (MAE and RMSE in A).")
ex = pd.read_csv(os.path.join(OUT, "tables", "test_terminals_extended.csv")); kc2 = ex[ex.terminal == "KCL_sum"]; ex = ex[ex.terminal != "KCL_sum"]
T(ex[["case", "terminal", "r2_signed", "r2_log10_floor1e30", "r2_log10_floor1e15", "median_rel_err_pct", "p95_rel_err_pct", "sign_acc_pct"]], size=7.5,
  cap="Table 8. Test split: R2 on the signed current and on log10|I| (floors 1e-30 A and 1e-15 A), relative errors and sign accuracy per terminal.")
Pp(f"R2 on the signed scale is dominated by the largest currents and is 0.9994-0.9999 for every terminal that carries a physical current; for terminals that are numerically zero (the gate current of PMOS OFF, 1e-16 A and below) it is not meaningful (-0.0002). R2 on log10|I| is penalised by labels below the floor and is therefore shown for two floors. "
   f"The sum of the four predicted terminal currents satisfies charge conservation by construction (median |sum|/max|I| = {kc2.median_rel_err_pct.max():.1e} %).")
for c in CASES: IMG(os.path.join(OUT, "plots", f"pred_vs_spice_{c}.png"), 16.5, f"Figure. Predicted versus SPICE |I| on the test split, {c} (red: sign mismatch; values below 1e-20 A are clipped for display).")
Pp("Terminals that are numerically zero (I_G of OFF devices, 1e-23 to 1e-30 A) are reproduced in log magnitude but carry no physical weight; their relative error is determined by the 1 fA floor. Because I_S is derived from charge conservation, its relative error exceeds that of I_D for OFF devices where |I_D| is comparable with |I_B|; "
   "the gate leakage depends on I_S + I_B = -(I_D + I_G), which does not suffer from this.")

# ------------------------------------------------------------------ 9 gates
H("9. CMOS Gate Design and SPICE Validation")
Pp("The gates are standard static CMOS: INV; NAND2 (series NMOS stack with internal node x between MnA and MnB); NOR2 (series PMOS stack with internal node y); AND2 = NAND2 followed by INV and OR2 = NOR2 followed by INV, each simulated as one complete transistor netlist. "
   "Input A drives the NMOS adjacent to the output node (NAND2) and the PMOS adjacent to VDD (NOR2); this assignment of pins to stack positions is a convention. All gates use Wn = 90 nm and Wp = 180 nm, within the trained width range.")
Pp("Corners. A corner is one draw of VDD ~ U(0.9, 1.1) V, T ~ U(-40, 125) C, one NMOS process draw shared by all NMOS devices and one PMOS draw shared by all PMOS devices (seed 303). The 200 corners are shared by all gates and by the parity checker. 18 gate/input cases x 200 corners = 3,600 records; for AND2 and OR2 both internal nodes (stack node and first-stage output) and the output voltage are recorded.")
Pp("Reference acceptance. A SPICE operating point is accepted using criteria that involve no ML model: (1) finite simulator output; (2) the output voltage within 5 % of VDD of the Boolean value; (3) positive static power; (4) an independent node-KCL check, in which every transistor is re-simulated alone at the recorded terminal voltages and the sum of the absolute node-KCL residuals over all non-source nodes is below 1e-3 of T. The solver protocol and its validation are described in Section 17.")

# ------------------------------------------------------------------ 10 gate prediction
H("10. Gate-Level Leakage Prediction")
Pp("For every gate record, the SPICE-recorded node voltages (stack nodes, stage outputs, output) and the exact logic inputs (0 or VDD) determine the terminal voltages (Vd, Vg, Vs, Vb) of every transistor; gate and body voltages select the case model (e.g. an NMOS with Vg = VDD uses NMOS ON). "
   "The case model predicts (I_D, I_G, I_S, I_B) from VDD, T, W, Vd, Vs and the process parameters of the corner, and T and S are formed by the terminal sums of Section 4. No SPICE current enters the prediction.")
Pp(f"Validation of the reconstruction. If the ML models are replaced by single-device SPICE simulations at the recorded voltages, S and T of the 3,600 records are reproduced with median relative error {orc['S']['median']:.1e} and {orc['T']['median']:.1e} (99th percentile {orc['S']['p99']:.1e} and {orc['T']['p99']:.1e}, maximum {orc['S']['max']:.1e} and {orc['T']['max']:.1e}); "
   "the gate-level errors reported below are therefore transistor-model errors. As a supplementary consistency test, the stack-node voltage was also predicted by the ML models alone, by solving Kirchhoff's current law at the node with bisection (columns ml_stack_node and spice_stack_node of the per-point file); the median absolute difference to SPICE is below 1.1 mV for all cases (out/gate/gate_stack_node_check.csv).")

# ------------------------------------------------------------------ 11 gate results
H("11. Gate-Level Results and Error Analysis")
cols = ["gate", "a", "b", "spice_median_A", "ml_median_A", "bias_pct", "median_err_pct", "p95_err_pct", "mean_accuracy_pct"]
T(gT[cols], size=7.5, cap="Table 9. Gate results for quantity T (3,600 points; 200 corners per row); bias = median signed relative error.")
T(gS[cols], size=7.5, cap="Table 10. Gate results for quantity S (VDD-rail current).")
Pp(f"Overall for T: mean accuracy {gM['T']['mean_accuracy_pct']:.2f} %, median error {gM['T']['median_err_pct']:.2f} %, 95th percentile {gM['T']['p95_err_pct']:.2f} %, bias {gM['T']['bias_pct']:.2f} %. Overall for S: mean accuracy {gM['S']['mean_accuracy_pct']:.2f} %, median {gM['S']['median_err_pct']:.2f} %, 95th percentile {gM['S']['p95_err_pct']:.2f} %.", bold=True)
best = gT.loc[gT.mean_accuracy_pct.idxmax()]; worst = gT.loc[gT.mean_accuracy_pct.idxmin()]; bl = lambda r: f"{r.gate} {int(r.a)}{'' if pd.isna(r.b) or str(r.b) == '' else int(r.b)}"
Pp(f"The most accurate case is {bl(best)} ({best.mean_accuracy_pct:.2f} %) and the least accurate is {bl(worst)} ({worst.mean_accuracy_pct:.2f} %). The spread between gates and input vectors is small because the error is dominated by the common transistor-model error.")
IMG(os.path.join(OUT, "gate", "gate_parity_T.png"), 7.5, "Figure. ML versus SPICE gate leakage T on logarithmic axes, all 3,600 gate simulations.")
IMG(os.path.join(OUT, "gate", "gate_parity_S.png"), 7.5, "Figure. ML versus SPICE gate leakage S on logarithmic axes.")
T(wg.head(8)[["kind", "a", "b", "corner", "temp", "vdd", "nmos_vth0", "pmos_vth0", "i_tot_vdd", "ml_T", "e"]], size=7.5, cap="Table 11. The eight gate records with the largest relative error e of T.")
T(dev[["model", "T_mean_acc", "T_median_err", "T_p95_err", "S_mean_acc", "S_median_err", "S_p95_err"]], size=8, cap="Table 12. Gate-level comparison of the best classical family (Histogram Gradient Boosting) and the selected structured models on 200 development corners that are disjoint from the 200 evaluation corners.")

# ------------------------------------------------------------------ 12 architecture
H("12. 8-Bit Parity Checker Architecture")
Pp("The parity checker is a three-level tree of seven XOR2 cells. Each XOR2 is built from the five gates of Part B as XOR2(a, b) = AND2(NAND2(a, b), OR2(a, b)), giving 7 x (NAND2 + OR2 + AND2) = 21 gate instances and 7 x 16 = 112 transistors (Wn = 90 nm, Wp = 180 nm). "
   "Level 1 combines p0..p7 in pairs, level 2 the four level-1 outputs and level 3 the two level-2 outputs. The Boolean model of the netlist reproduces the parity function for all 256 input vectors. The complete netlist is simulated once per (corner, vector): "
   f"{len(cs):,} operating points. They are accepted with the criteria of Section 9 (all 21 gate outputs within 5 % VDD of their Boolean values); {int(cs.ok.sum()):,} of {len(cs):,} were accepted, {int((cs.attempt==0).sum()):,} at the assignment's solver options and {int((cs.attempt>0).sum()):,} after the relaxation steps of Section 17; the largest KCL residual metric is {cs.kcl_rel.max():.2e}.")

# ------------------------------------------------------------------ 13 hierarchical prediction
H("13. Hierarchical Circuit-Level Prediction")
Pp("For each input vector the logic value at the two inputs of every gate instance is obtained by Boolean simulation; the Part-B ML prediction for that (gate type, input combination, corner) is looked up and the 21 values are summed. The prediction function receives only the input vector, the corner identifier and the table of Part-B predictions; "
   "no node voltage, device state or current of the complete-circuit simulation enters it (this is checked statically and by recomputation in the validation suite). The Part-B predictions use the isolated-gate SPICE node voltages as prescribed by the assignment. The method assumes that every gate input is at exactly 0 or VDD and ignores loading between gates.")

# ------------------------------------------------------------------ 14 circuit results
H("14. Circuit-Level Results and Error Analysis")
T(cT[["group", "key", "n", "spice_median_A", "ml_median_A", "bias_pct", "median_err_pct", "p95_err_pct", "mean_accuracy_pct"]], size=7.5, cap="Table 13. Parity checker, quantity T: overall and by Hamming weight, output parity, temperature and VDD bin.")
Pp(f"Overall (51,200 points) for T: mean accuracy {cM['T']['mean_accuracy_pct']:.2f} %, median error {cM['T']['median_err_pct']:.2f} %, 95th percentile {cM['T']['p95_err_pct']:.2f} %, bias {cM['T']['bias_pct']:.2f} %. For S: plain sum {cM['S']['mean_accuracy_pct']:.2f} % (bias {cM['S']['bias_pct']:.1f} %), load-corrected {cM['S_load_corrected']['mean_accuracy_pct']:.2f} %.", bold=True)
IMG(os.path.join(OUT, "circuit", "circuit_parity_T.png"), 7.5, "Figure. Hierarchical ML versus complete-circuit SPICE leakage T on logarithmic axes (colour: temperature).")
IMG(os.path.join(OUT, "circuit", "circuit_error_analysis.png"), 16.5, "Figure. Signed relative error of T versus temperature, versus VDD, and its histogram.")
H("14.1 Per-input-vector circuit leakage estimation results", 2)
Pp("The 51,200 circuit records were aggregated by input vector: for each of the 256 vectors, the 200 PVT records give the statistics below (no additional simulation). The full 256-row tables are provided as out/circuit/circuit_results_per_vector_T.csv, _S.csv and _S_loadcorrected.csv with the columns vector, input_bits, parity, hamming_weight, pvt_count, spice_median, ml_median, mean_spice, mean_ml, bias_pct, mean_signed_err_pct, median_error_pct, p95_error_pct, max_error_pct, mean_mape_pct, mean_accuracy_pct and quantity. "
   "Definitions are those of the other tables: relative error e = (ML - SPICE)/SPICE; bias_pct is 100 x the median of the signed relative error (the mean signed error is given separately); median and p95 refer to |e|; mean_mape_pct = 100 x mean|e| and mean_accuracy_pct = 100 - mean_mape_pct.")
rows = []
for nm_, d_ in (("T", pvT), ("S plain sum", pvS), ("S load-corrected", pvC)):
    rows.append([nm_, len(d_), int((d_.pvt_count == 200).sum()), d_.median_error_pct.min(), d_.median_error_pct.median(), d_.median_error_pct.max(), d_.p95_error_pct.min(), d_.p95_error_pct.max(), d_.mean_accuracy_pct.min(), d_.mean_accuracy_pct.max(), 100 - d_.mean_mape_pct.mean()])
T(pd.DataFrame(rows, columns=["quantity", "vectors", "with 200 PVT", "min median err %", "median of median err %", "max median err %", "min p95 %", "max p95 %", "min mean acc %", "max mean acc %", "mean of vector accuracies %"]), size=7, cap="Table 14. Summary of the three 256-row per-vector tables; the last column reproduces the global 51,200-point mean accuracy.")
vc = ["vector", "input_bits", "parity", "hamming_weight", "spice_median", "ml_median", "bias_pct", "median_error_pct", "p95_error_pct", "mean_accuracy_pct"]
T(pvT.nsmallest(5, "median_error_pct")[vc], size=7.5, cap="Table 15. The five most accurate input vectors (T, lowest median error).")
T(pvT.nlargest(5, "median_error_pct")[vc], size=7.5, cap="Table 16. The five least accurate input vectors (T, highest median error).")
IMG(os.path.join(OUT, "circuit", "circuit_per_vector_median_error.png"), 16.5, "Figure. Per-vector median absolute relative error of T (256 vectors in vector order; orange: odd parity).")
IMG(os.path.join(OUT, "circuit", "circuit_per_vector_p95_error.png"), 16.5, "Figure. Per-vector 95th-percentile absolute relative error of T.")
IMG(os.path.join(OUT, "circuit", "circuit_per_vector_by_hamming.png"), 7, "Figure. Per-vector median error grouped by Hamming weight.")
Pp(f"Across the 256 vectors the median error of T lies between {an['median_error_range_pct'][0]:.2f} % and {an['median_error_range_pct'][1]:.2f} % and the 95th-percentile error between {an['p95_error_range_pct'][0]:.2f} % and {an['p95_error_range_pct'][1]:.2f} %. "
   f"Output parity shows no detectable association with the error (median of the per-vector median errors {an['median_of_median_error_by_parity']['0']:.3f} % for even and {an['median_of_median_error_by_parity']['1']:.3f} % for odd parity; Mann-Whitney p = {an['mannwhitney_parity_median_error_p']:.2f}). "
   f"Hamming weight is positively associated with the error (Spearman rho = {an['spearman_hamming_vs_median_error'][0]:.2f} for the median and {an['spearman_hamming_vs_p95_error'][0]:.2f} for the 95th-percentile error); the ten worst vectors have Hamming weights {sorted(an['top10_worst_median_error_hamming'])}. "
   "This is an association only. Vectors with more ones place more gate inputs at VDD, and T includes the gate currents of those inputs; whether that explains the trend was not tested. The vector-to-vector differences are a few tenths of a percentage point.")
H("14.2 Composition error", 2)
Pp("The circuit prediction is the sum of gate predictions, ML_circuit = Σ_g ML_g. Let G denote the sum of the isolated-gate SPICE leakages of the same 21 instances. Then (1 + e_total) = (1 + e_ML)(1 + e_comp) holds exactly, with e_ML = ML_circuit/G - 1 (transistor-model and stack-modelling error) and e_comp = G/SPICE_circuit - 1 (composition: loading, non-ideal gate-input levels, DC differences). The decomposition is multiplicative, not additive.")
dec = []
for q, nm_ in (("T", "T"), ("S", "S, plain sum"), ("S_load_corrected", "S, load-corrected")):
    kml = f"{q}_ml_vs_gate_spice_sum" if q != "S_load_corrected" else "S_load_corrected_ml_vs_gate_spice_sum"; kco = f"{q}_gate_spice_sum_vs_circuit" if q != "S_load_corrected" else "S_load_corrected_gate_spice_sum_vs_circuit"
    dec.append([nm_, cM[q]["mean_err_pct"], cM[kml]["mean_err_pct"], cM[kml]["median_err_pct"], cM[kco]["mean_err_pct"], cM[kco]["median_err_pct"], cM[kco]["bias_pct"]])
T(pd.DataFrame(dec, columns=["quantity", "total mean err %", "ML-vs-gate-sum mean %", "ML-vs-gate-sum median %", "composition mean %", "composition median %", "composition bias %"]), size=8, cap="Table 17. Decomposition of the circuit error (51,200 points).")
Pp("For T and for load-corrected S the composition error is of the order of 1e-2 %: loading between gates and non-ideal gate-input levels are negligible for this circuit, and the circuit error equals the gate-level model error. For plain-sum S the composition term is a systematic bias, explained in Section 16.")

# ------------------------------------------------------------------ 15 PVT
H("15. PVT Sensitivity and Robustness Analysis")
T(rb[rb.group.isin(["temperature_C", "VDD_V"])][["group", "key", "n", "bias_pct", "median_err_pct", "p95_err_pct", "mean_accuracy_pct"]], size=7.5, cap="Table 18. Gate error (quantity T) by temperature and VDD bin.")
T(rb[rb.group.str.contains("quartile")][["group", "key", "n", "bias_pct", "median_err_pct", "p95_err_pct", "mean_accuracy_pct"]], size=7, cap="Table 19. Gate error (quantity T) by process-parameter quartile.")
_t = rb[rb.group == "temperature_C"]; _v = rb[rb.group == "VDD_V"]
Pp(f"The mean accuracy is between {spread_all.min():.2f} % and {spread_all.max():.2f} % in every temperature, VDD, vth0 and toxe bin (Tables 13, 18, 19). The median error of T increases mildly with temperature ({_t.median_err_pct.iloc[0]:.2f} % in the coldest bin, {_t.median_err_pct.iloc[-1]:.2f} % in the hottest) and is highest in the lowest VDD bin ({_v.median_err_pct.iloc[0]:.2f} %). "
   f"The largest single gate error is {100*float(wg.e.abs().max()):.1f} % (Table 11; {wg.temp.iloc[0]:.0f} C) and the largest circuit error is {100*float(cw.err_T.abs().max()):.1f} %; {int((cw.corner==cw.corner.mode()[0]).sum())} of the 100 largest circuit errors belong to a single corner (corner {int(cw.corner.mode()[0])}, T = {cw.temp.iloc[0]:.0f} C, VDD = {cw.vdd.iloc[0]:.3f} V). "
   "Errors are therefore correlated within a corner: when a corner is mispredicted, all gates at that corner are mispredicted in the same direction, and summing 21 gates does not reduce this error.")
Pp("Generalisation. The gate and circuit corners are random draws distinct from the training samples, and the evaluation corners are distinct from the 200 development corners used for design decisions, so the corner dimension (VDD, T, process) is evaluated out of sample. "
   "The terminal-voltage dimension is covered by the supplementary gate-corner samples that are part of the method; the results are not claimed to extend to other bias conditions, gate sizes or technologies.")

# ------------------------------------------------------------------ 16 T and S
H("16. Leakage-Current Accounting: T and S")
Pp(f"T and S were defined in Section 4. They were verified from the raw voltage-source currents: T x VDD equals the sum of V_k I_k over all sources (maximum relative difference {dv['gate_T_equals_static_power_over_VDD_max_rel_diff']:.1e} for gates and {dv['circuit_T_equals_static_power_over_VDD_max_rel_diff']:.1e} for the circuit), "
   f"and the currents delivered by VDD and the inputs balance the Vss current to within {100*dv['circuit_source_balance_max_rel']:.2f} % of T. S differs from T by more than 1 % in {100*dv['gate_cases_with_S_ne_T_gt_1pct']:.0f} % of the gate records (median S/T = {dv['gate_S_over_T_median']:.3f}).")
Pp("In the isolated gate, the gate current of a device whose gate is held high is supplied by the input source and therefore is part of T but not of S. In the complete circuit, the same current flows into a load input from the driving gate, i.e. from the VDD rail through the driver's pull-up path, and is part of S. "
   "Consequently the plain sum of isolated-gate S values omits the gate currents of loads on internal nets held at VDD. The load-corrected estimate adds, for every gate input that is an internal net at logic 1, the predicted gate current of that input (from the same transistor models), S_circuit = Σ_g [ S_g + 1(a internal and high) · I_in,a,g + 1(b internal and high) · I_in,b,g ]. "
   f"With SPICE gate data alone this identity reproduces the complete-circuit S to a mean error of {dv['load_correction_SPICE_only_mean_abs_err_pct']:.3f} %, whereas the plain sum has a mean error of {dv['plain_gate_sum_S_mean_abs_err_pct']:.1f} %; for T the isolated-gate sum has a mean error of {dv['T_gate_sum_mean_abs_err_pct']:.3f} %.")
_D = pd.read_csv(os.path.join(OUT, "circuit", "circuit_results_per_point.csv.gz")); _e = (_D.gsum_S - _D.i_vdd) / _D.i_vdd
Pp(f"The bias of the plain-sum S depends on temperature (median {100*_e[_D.temp<-10].median():.1f} % below -10 C and {100*_e[_D.temp>80].median():.1f} % above 80 C), as expected because gate tunnelling is almost temperature independent while sub-threshold leakage grows with temperature, so the omitted gate-current term is a larger fraction of S at low temperature.")
T(pd.DataFrame([["T (total VDD-level source current)", gT_all["mean_accuracy_pct"], cM["T"]["mean_accuracy_pct"]], ["S, plain sum of isolated-gate predictions", gM["S"]["mean_accuracy_pct"], cM["S"]["mean_accuracy_pct"]], ["S, load-corrected", np.nan, cM["S_load_corrected"]["mean_accuracy_pct"]]],
   columns=["quantity", "gate-level mean accuracy %", "circuit-level mean accuracy %"]), size=8.5, cap="Table 20. Mean accuracy for the three leakage quantities. The 99 % level at circuit level holds for T and for load-corrected S, not for the plain-sum S.")
Pp("Interpretation of 'circuit overall leakage current'. The assignment text does not state whether the rail current alone or all source current is meant. The lecture's gate-leakage equation adds the sub-threshold, body and gate leakage of every transistor, which corresponds to T. Because the interpretation is not settled, all three results are reported, and the circuit accuracy must be quoted together with the quantity it refers to.")

# ------------------------------------------------------------------ 17 numerical validation
H("17. Numerical Validation and SPICE Convergence")
Pp("Solver protocol. Level 0 is the assignment's option set (gmin = 1e-18, abstol = 1e-18, reltol = 1e-6, vntol = 1e-9). For operating points on which ngspice does not converge or the result fails the KCL check, only the convergence tolerances are relaxed in six deterministic steps (abstol 1e-16, abstol 1e-15, reltol 1e-4, reltol 1e-3, then stepping options and a logic-level initial guess); "
   "the strict tolerances are numerically unreachable for some BSIM4 operating points. When ngspice fails to converge it prints the last iterate, which the independent KCL check detects. The accepted result is always checked with the criteria of Section 9; none depends on the ML models.")
T(lv[["rung", "n", "median_rel_diff", "p95_rel_diff", "p99_rel_diff", "max_rel_diff", "median_dVout_V", "p95_dVout_V", "max_dVout_V"]], size=7.5,
  cap="Table 21. Relaxed level (3: abstol 1e-15, reltol 1e-4; 4: reltol 1e-3) versus strict level 0 on sampled gate points where both converge: relative difference of I(VDD) and absolute difference of V(out).")
Pp(f"Outcome. Gate references: {int(gstat.ok.sum())}/{len(gstat)} accepted, {int((gstat.attempt==0).sum())} at level 0 and {int((gstat.attempt>0).sum())} after relaxation (largest residual metric {gstat.kcl_rel.max():.2e}); circuit references: {int(cs.ok.sum()):,}/{len(cs):,} accepted, {int((cs.attempt==0).sum()):,} at level 0 and {int((cs.attempt>0).sum()):,} after relaxation. "
   "The model card also produces the ngspice message 'toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored' whenever toxe is swept; it is a parameter-consistency note, not a convergence message.")
H("17.1 Warning-bearing gate references", 2)
Pp(f"{len(sw)} accepted gate references ({int((sw.kind=='AND2').sum())} AND2, {int((sw.kind=='OR2').sum())} OR2) carry an ngspice failure message. In each log, dynamic gmin stepping, true gmin stepping and source stepping report failure and ngspice then finishes with its transient-op fallback ('Transient op finished successfully'). "
   f"{int((sw.accepted_level==0).sum())} of these records were accepted with the assignment's solver options (strict-setting records that converged through ngspice's own fallback) and {int((sw.accepted_level>0).sum())} at a relaxed level (documented fallback configuration). "
   f"Each operating point was re-solved using up to 15 alternative solver configurations and initial guesses, yielding {int(nv.min())}-{int(nv.max())} KCL-valid solutions per point, with a median of {int(nv.median())}; each was checked for logic level, power sign and KCL. "
   f"Classification (fixed criteria, independent of the ML models): {'; '.join(f'{k}: {v}' for k, v in cls_.items())}; numerically questionable: {int((sw.classification=='numerically_questionable').sum())}; invalid: {int((sw.classification=='invalid').sum())}. "
   f"The largest deviation of any KCL-valid re-solve from the stored reference is {sw.resolve_max_rel_diff.max():.1e} and the largest KCL residual metric is {sw.kcl_rel.max():.1e}. The per-record results and class counts are in out/audit/solver_warning_audit.csv and solver_warning_summary.csv. Raw solver messages were not stored for the 51,200 circuit references, so no equivalent per-record audit exists for them.")
Pp(f"Repeatability. Each warning-bearing record was simulated twice at its accepted solver level and the runs were compared quantity by quantity ({int(r12.n_quantities.min())} quantities per run: source currents, every recorded node voltage, the terminal voltages and SPICE terminal currents of all transistors, and T and S computed in two ways; out/audit/solver_repeatability_detail.csv). "
   f"The two runs were exactly equal in all quantities for {int(r12.all_exactly_equal.sum())} of {len(r12)} records (maximum absolute difference {r12.max_abs_diff.max():.1e}, maximum relative difference {r12.max_rel_diff.max():.1e}). The comparison with the stored generation record covers the {int(rst.n_quantities.min())} quantities that the stored file contains (it has no input-source currents or derived terminal voltages): "
   f"{int(rst.all_exactly_equal.sum())} of {len(rst)} records are exactly equal. The tolerance applied is |difference| <= abs_tol + 1e-9 x magnitude with abs_tol = 1e-21 A for currents and 1e-12 V for voltages; all comparisons pass. Equality is claimed for these quantities only, on the machine and ngspice build used.")
H("17.2 DC-branch investigation", 2)
Pp(f"To test for multiple DC solutions, {len(dcp)} cold-corner gate points (T < -20 C) were re-solved from 8 random initial guesses at three relaxed solver levels ({len(dcd)} attempts). Each finite solution was checked for logic level, positive power, KCL, repeatability under a repeated run and consistency of the internal-node voltages with the reference solution (out/audit/dc_branch_probe_detail.csv). "
   f"Classification of the attempts: {'; '.join(f'{k}: {v}' for k, v in dcs.items())}. All questionable attempts are finite iterates that fail the KCL check; none failed the logic or power test, and all repeated runs were identical. "
   f"{int((dcp.n_valid>0).sum())} of {len(dcp)} points produced at least one valid and physically consistent solution. Among valid solutions of one point the supply current varies by at most {dcp.spread_pct.max():.2f} %, the largest internal-node difference to the reference is {1e3*dcp.max_node_dV_vs_reference.max():.2f} mV, and {int(dcp.n_distinct_branch_candidates.sum())} valid solutions differ from the reference by more than 1 % in T or 10 mV in a node. "
   f"{len(nov)} probes ({'; '.join(f'{r.kind} {r.ab} at corner {r.corner}' for r in nov.itertuples())}) produced no valid solution from the tested initialisations.")
Pp("Among the solutions successfully examined, no substantial alternative valid DC branch was found. This is a limited multi-start experiment; it does not establish that the DC operating point is globally unique.")
H("17.3 Reproducibility of the numerical results", 2)
if rc:
    gs, cs_ = rc["gate_spice"], rc["circuit_spice"]; cm = rc["circuit_metrics"]; gm = rc["gate_metrics"]
    Pp(f"The SPICE stages were re-executed in an independent directory with the same seeds and compared with the stored results: corner definitions identical ({rc['corners_identical']}); gate references (3,600 records) maximum relative difference of I(VDD) {gs['i_vdd_rel_diff']['max']:.1e}; circuit references ({cs_['n_new']:,} records) maximum relative difference of T {cs_['i_tot_rel_diff']['max']:.1e}; "
       f"base transistor datasets identical in the inputs, with currents differing by at most {max(v['currents_rel_diff']['max'] for k, v in rc.items() if k.startswith('base_')):.1e} relative (print precision). A full retraining of the models in the same run gave a gate T accuracy of {gm['rerun']['T']['mean_accuracy_pct']:.2f} % and a circuit T accuracy of {cm['T']['rerun']:.2f} % "
       f"(selected models: {gm['stored']['T']['mean_accuracy_pct']:.2f} % and {cm['T']['stored']:.2f} %), which indicates the run-to-run variation of the training stage; the selected models were not changed.")

# ------------------------------------------------------------------ 18 limitations
H("18. Limitations and Future Improvements")
B(["The circuit-level accuracy depends on the leakage quantity: T and load-corrected S exceed 99 %, the plain sum of isolated-gate S gives " + f"{cM['S']['mean_accuracy_pct']:.2f} %. The interpretation intended by the assignment is not settled.",
   f"Worst-case errors exceed the mean considerably: {100*float(wg.e.abs().max()):.1f} % for a gate record and {100*float(cw.err_T.abs().max()):.1f} % for a circuit record; the mean accuracy varies between {spread_all.min():.2f} % and {spread_all.max():.2f} % across PVT bins.",
   "Errors are correlated within a PVT corner and are not reduced by summing gates; additional training data near extreme process corners and larger ensembles are the most direct remedies.",
   "The models are trained on isolated single-device sweeps with fixed gate and body voltages; intermediate gate voltages are not modelled, and gate inputs are assumed to be exactly 0 or VDD. Gate widths are fixed (90/180 nm), and a single technology, supply and circuit were studied.",
   "Part B uses SPICE-recorded node voltages as prescribed; the ML-only stack-node solution is a supplementary consistency test.",
   "Relative errors of terminal currents below about 1 fA are meaningless because of floating-point noise in the SPICE labels; a 1 fA floor is used.",
   f"{int((gstat.attempt>0).sum())} gate and {int((cs.attempt>0).sum()):,} circuit references needed relaxed solver tolerances, and {len(sw)} gate references needed ngspice's transient-op fallback or relaxed tolerances; they were validated individually (Section 17), but per-record solver messages of the circuit references were not stored. The DC-branch study is a limited multi-start experiment.",
   "The classical model families were trained on an 80,000-row subsample of the extended data. Design decisions (feature set, supplementary sampling, epochs) used 200 development corners, which are not statistically independent of those choices.",
   "The dependence of the error on Hamming weight is an association that was not investigated mechanistically."])

# ------------------------------------------------------------------ 19 conclusion
H("19. Conclusion")
Pp(f"Models of single 45 nm LP transistors, trained on SPICE data and structured so that charge conservation and the Vds = 0 behaviour are represented explicitly, predict the total static leakage T of INV, NAND2, NOR2, AND2 and OR2 with {gT_all['mean_accuracy_pct']:.2f} % mean accuracy (median error {gT_all['median_err_pct']:.2f} %) over 3,600 SPICE references, "
   f"and the leakage of the 8-bit parity checker with {cT_all['mean_accuracy_pct']:.2f} % mean accuracy (median error {cT_all['median_err_pct']:.2f} %, 95th percentile {cT_all['p95_err_pct']:.2f} %) over 51,200 complete-netlist references; the gate-level predictions use the node voltages recorded from gate-level SPICE operating points as prescribed, and the circuit-level prediction uses no complete-circuit SPICE data as input. "
   f"The 99 % target is therefore met under the declared metric for T and for load-corrected S ({cM['S_load_corrected']['mean_accuracy_pct']:.2f} %); it is not met by the plain sum of isolated-gate rail currents ({cM['S']['mean_accuracy_pct']:.2f} %), whose deficit is an accounting effect that can be quantified and corrected. "
   "Composition of gate predictions is accurate for this circuit; the remaining error is the transistor-model error at the edges of the PVT and process space.")

# ------------------------------------------------------------------ 20 reproducibility, references
H("20. Reproducibility, References and Acknowledgements")
B([f"Seeds: base samples 101-104, split 2024, corners 303 (development corners 777), ML 404, supplementary data 1-3. Model-card sha256 {env['model_card_sha256']}.",
   f"Software: ngspice-47 (external executable), python {env['python']}, scikit-learn {env['sklearn']}, numpy {env['numpy']}, pandas {env['pandas']}, torch {env['torch']} (training only; inference uses numpy).",
   "run_all.sh executes the stages environment check, model-card hash, corners, data generation, data check, training, evaluation, gate and circuit SPICE and ML, error decomposition, solver audit, netlists, report, packaging and fresh-extraction validation, logs each stage and stops at the first failure. validate_all.py recomputes the principal metrics from the raw CSV files and fails on any mandatory check that is not satisfied; each package contains a SHA-256 manifest.",
   "All paths are relative to DVD_ROOT; ngspice is located through PATH or the NGSPICE environment variable."])
Pp("Project repository", bold=True)
_p = Pp("The complete project source code, trained model artifacts, simulation netlists, datasets, validation results and reproducibility instructions are available in the accompanying GitHub repository: ")
LINK(_p, "https://github.com/shouryasarkar713/45nm-lp-pvt-leakage-estimation")
Pp("References", bold=True)
B(["BSIM Group, BSIM4 MOSFET Model - User's Manual, University of California, Berkeley.",
   "Predictive Technology Model (PTM), Arizona State University, 45 nm low-power bulk model card.",
   "ngspice: open-source mixed-level, mixed-signal circuit simulator, version 47.",
   "F. Pedregosa et al., Scikit-learn: Machine Learning in Python, Journal of Machine Learning Research 12, 2011.",
   "A. Paszke et al., PyTorch: An Imperative Style, High-Performance Deep Learning Library, NeurIPS 2019.",
   "K. Roy, S. Mukhopadhyay, H. Mahmoodi-Meimand, Leakage current mechanisms and leakage reduction techniques in deep-submicrometer CMOS circuits, Proc. IEEE 91(2), 2003."])
Pp("Statement on AI assistance. AI tools were used to assist with software development and report drafting.", italic=True, size=9.5)
