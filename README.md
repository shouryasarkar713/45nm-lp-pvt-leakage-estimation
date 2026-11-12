# ML-Driven PVT-Variation-Aware Leakage Estimation - 45 nm LP, VDD = 1 V

Repository: https://github.com/shouryasarkar713/45nm-lp-pvt-leakage-estimation

Digital VLSI Design course project. Machine-learning models of single 45 nm LP transistors, trained on SPICE data, are used to estimate the static leakage of five CMOS gates and, by hierarchical composition, of an 8-bit parity checker under process, voltage and temperature (PVT) variation.
The technical report is available as [PDF](report/Project1_Report_45nmLP_1V_ParityChecker.pdf) and [DOCX](report/Project1_Report_45nmLP_1V_ParityChecker.docx).

## 1. Technical objectives
* **Part A** - characterise single NMOS/PMOS transistors (ON and OFF) over random PVT draws and train four terminal-current models (I_D, I_G, I_S, I_B) per case.
* **Part B** - apply the trained models to INV, NAND2, NOR2, AND2 and OR2 (18 input cases x 200 PVT corners), using the internal-node and output voltages recorded from the gate-level SPICE operating points, and compare with the gate SPICE leakage.
* **Part C** - compose the gate predictions hierarchically for an 8-bit parity checker (256 input vectors x 200 corners) and compare with complete-netlist SPICE; the complete-circuit SPICE results are used only as the evaluation reference.

## 2. Technology and simulation configuration
| Item | Value |
|---|---|
| Model card | `model/45nm_LP.pm` (PTM low-power 45 nm, BSIM4), sha256 `e2423b27f22e994cdf8aeaa116c25114ee5166c19e35e7bae4a3a03214d512be` |
| Supply / temperature | VDD 1.0 V sampled in 0.9-1.1 V; -40 to 125 C |
| Devices | L = 45 nm; transistor data W = 45-360 nm; gates and circuit Wn = 90 nm, Wp = 180 nm |
| Process parameters | vth0, toxe, ndep, u0, xj within +-10 % of nominal, independent NMOS/PMOS |
| Simulator | ngspice-47, `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9` |
| Software | Python 3.14.6, numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, torch 2.15.0+cpu (training only) |

## 3. Methodology
* **Transistor-level modelling.** Four single-FET datasets of 10,000 rows (70/15/15 split) plus supplementary SPICE training data. Four classical families (Ridge, Random Forest, Histogram Gradient Boosting, MLP) are compared with a physics-structured MLP that writes I_D = id0 + sgn(Vds)|Vds|G, learns I_G and I_B directly and derives I_S from charge conservation. The structured MLP is selected for all four cases (test median relative error 0.67-1.08 %).
* **Gate-level estimation.** Terminal voltages of every transistor come from the SPICE-recorded node voltages; the case model predicts the terminal currents and the leakage is formed by summing the currents of the terminals tied to VDD.
* **Hierarchical parity-checker estimation.** The parity checker is seven XOR2 cells, each AND2(NAND2, OR2), i.e. 21 gate instances and 112 transistors. The prediction for a vector is the sum of the Part-B gate predictions for the logic values at the gate inputs.

## 4. Key results
Accuracy = 100 x (1 - MAPE), MAPE = mean |ML - SPICE| / |SPICE| over the evaluated operating points.

| Quantity | Mean accuracy | Median error | 95th-percentile error |
|---|---|---|---|
| Gate-level, T | 99.37 % | 0.44 % | 1.84 % |
| Gate-level, S | 99.31 % | 0.42 % | 2.33 % |
| Circuit-level, T | 99.48 % | 0.40 % | 1.36 % |
| Circuit-level, S (plain sum of gate predictions) | 80.63 % | 12.44 % | 60.16 % |
| Circuit-level, S (load-corrected) | 99.50 % | 0.38 % | 1.30 % |

The 256-row per-vector tables are `out/circuit/circuit_results_per_vector_T.csv`, `_S.csv` and `_S_loadcorrected.csv`.

## 5. Leakage quantities T and S
* **T** - total static current delivered by all voltage sources held at VDD (VDD rail plus inputs at logic 1) = static power / VDD.
* **S** - current drawn from the VDD rail alone.
* **S, load-corrected** - S with the gate current of internal nets at logic 1, supplied through the upstream drivers, added to the plain sum of isolated-gate predictions.

The plain-sum S differs from the circuit value because that gate current is supplied by the input source in the isolated gate but by the VDD rail in the circuit. The assignment does not state which quantity is meant; all three are reported and every result table carries a `quantity` column. Details are in Section 16 of the report.

## 6. Repository structure
| Path | Content |
|---|---|
| `model/45nm_LP.pm` | assigned model card |
| `data/base/` | four 10,000-row transistor datasets |
| `data/gate_spice_official.csv`, `data/corners_official.csv` | gate SPICE references (3,600 records) and the 200 PVT corners |
| `netlists/` | example transistor, gate and parity-checker netlists |
| `work/circuit_spice/` | complete-circuit SPICE references (200 corners x 256 vectors) |
| `src/` | all scripts |
| `out/data_check/` | Kirchhoff check and trend plots |
| `out/tables/`, `out/models/`, `out/plots/` | model comparison, selected models, test metrics and plots |
| `out/gate/`, `out/circuit/` | gate and circuit results, parity plots, per-vector tables |
| `out/audit/` | numerical validation: solver-warning audit, repeatability, DC-branch study |
| `out/validation_output.txt`, `MANIFEST_sha256.txt` | validation output and SHA-256 manifest |
| `report/` | technical report (PDF, DOCX) |
| `run_all.sh`, `requirements.txt` | execution script and dependencies |

## 7. Installation
Python 3.14 with the packages in `requirements.txt` (torch is needed only for retraining) and ngspice-47 on `PATH` (or `NGSPICE=/path/to/ngspice`, with `NGSPICE_LD_LIBRARY_PATH` for a bundled library directory).
```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```
`DVD_ROOT` may select the project root (default: the parent of `src/`). The selected models run with numpy only.

## 8. Execution
```
bash run_all.sh                                  # 20 logged stages, stops at the first failure
bash run_all.sh --reuse                          # reuse stages with a completion marker
bash run_all.sh --skip "06_train 11_circuit_spice"
```
Stages: environment check, model-card hash, corners, base and supplementary data, data check, training, test evaluation, gate SPICE, gate ML evaluation, circuit SPICE, circuit ML evaluation, error decomposition, solver audit, netlists, report, package build, fresh-extraction validation. Typical cost on 16 cores: data about 15 min, training 1-2 h, circuit SPICE 40-60 min. The supplementary training data are regenerated by stage 04b.

## 9. Validation
```
cd src
python validate_all.py        # exit code 1 on any failed check
sha256sum -c MANIFEST_sha256.txt   # run from the repository root
```
`validate_all.py` recomputes the principal metrics from the raw CSV files, verifies the datasets, models, gate and circuit tables, the 256 per-vector tables, the solver audit, the report numbers and the manifest. One check (E6) re-simulates an example netlist and requires ngspice. The result of the validation run is stored in `out/validation_output.txt`.

## 10. Reproducibility
Seeds: base samples 101-104, split 2024, corners 303 (development corners 777), ML 404, supplementary data 1-3. SPICE stages are deterministic; training uses torch and is not bit-reproducible across runs.
The numerical validation (solver protocol, 27 gate references with solver warnings, repeatability and DC-branch studies) is documented in Section 17 of the report.

## 11. AI assistance
AI tools were used to assist with software development and report drafting.
