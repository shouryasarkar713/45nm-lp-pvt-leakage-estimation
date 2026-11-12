"""Deliverable 8: hierarchical ML prediction of the parity checker vs the complete-circuit SPICE reference."""
import os, json, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import *
from circuit_eval import *
from corners import load_corners

CD = os.path.join(OUT, "circuit"); os.makedirs(CD, exist_ok=True)

def stats(ref, ml):
    e = (ml - ref) / ref
    return dict(n=int(len(ref)), spice_median_A=float(np.median(ref)), ml_median_A=float(np.median(ml)), bias_pct=float(100 * np.median(e)),
                median_err_pct=float(100 * np.median(np.abs(e))), p95_err_pct=float(100 * np.quantile(np.abs(e), .95)),
                mean_err_pct=float(100 * np.mean(np.abs(e))), mean_accuracy_pct=float(100 * (1 - np.mean(np.abs(e)))))

def gate_dict(P, col, ref):
    d = {}
    for r in P.itertuples():
        a = int(r.a); b = int(r.b) if r.b >= 0 else 0
        d[(r.kind, a, b, int(r.corner))] = getattr(r, col)
    return d

if __name__ == "__main__":
    P = pd.read_csv(os.path.join(OUT, "gate", "gate_results_per_point.csv")); S = load_circuit_spice(); corners = sorted(S.corner.unique())
    assert len(S) == 51200
    hT = hierarchical_predict(gate_dict(P, "ml_T", None), corners).rename(columns={"ml": "ml_T"})     # Part-B ML gate predictions only
    hS = hierarchical_predict(gate_dict(P, "ml_S", None), corners).rename(columns={"ml": "ml_S"})
    gT = hierarchical_predict(gate_dict(P, "i_tot_vdd", None), corners).rename(columns={"ml": "gsum_T"})  # SPICE gate sums (diagnostic only)
    gS = hierarchical_predict(gate_dict(P, "i_vdd", None), corners).rename(columns={"ml": "gsum_S"})
    hSc = hierarchical_predict_S_loadcorrected(gate_dict(P, "ml_S", None), gate_dict(P, "ml_in_a", None), gate_dict(P, "ml_in_b", None), corners).rename(columns={"ml": "ml_S_loadcorr"})
    gSc = hierarchical_predict_S_loadcorrected(gate_dict(P, "i_vdd", None), gate_dict(P, "i_in_a", None), gate_dict(P, "i_in_b", None), corners).rename(columns={"ml": "gsum_S_loadcorr"})
    D = S.merge(gSc, on=["corner", "vec"]).merge(hSc, on=["corner", "vec"]).merge(hT, on=["corner", "vec"]).merge(hS, on=["corner", "vec"]).merge(gT, on=["corner", "vec"]).merge(gS, on=["corner", "vec"])
    C = load_corners("official").set_index("corner"); D["vdd"] = D.corner.map(C.vdd); D["temp"] = D.corner.map(C.temp)
    bits = np.array(VECS); D["hamming"] = bits[D.vec.values].sum(1); D["parity"] = D.hamming % 2
    D.to_csv(os.path.join(CD, "circuit_results_per_point.csv.gz"), index=False)
    M = dict(T=stats(D.i_tot.values, D.ml_T.values), S=stats(D.i_vdd.values, D.ml_S.values))
    # controlled decomposition  (1+e_total) = (1+e_ML-vs-gate-SPICE-sum) * (1+e_composition)   [exact, multiplicative]
    for q, (ref, ml, gs) in {"T": ("i_tot", "ml_T", "gsum_T"), "S": ("i_vdd", "ml_S", "gsum_S")}.items():
        M[f"{q}_ml_vs_gate_spice_sum"] = stats(D[gs].values, D[ml].values)       # transistor-ML + stack modelling error only
        M[f"{q}_gate_spice_sum_vs_circuit"] = stats(D[ref].values, D[gs].values)  # composition / loading / DC error only
    M["S_load_corrected"] = stats(D.i_vdd.values, D.ml_S_loadcorr.values)
    M["S_load_corrected_ml_vs_gate_spice_sum"] = stats(D.gsum_S_loadcorr.values, D.ml_S_loadcorr.values)
    M["S_load_corrected_gate_spice_sum_vs_circuit"] = stats(D.i_vdd.values, D.gsum_S_loadcorr.values)
    json.dump(M, open(os.path.join(CD, "circuit_overall_metrics.json"), "w"), indent=1)
    rows = [dict(group="ALL", key="all", **stats(D.i_tot.values, D.ml_T.values))]
    for k, s in D.groupby("hamming"): rows.append(dict(group="hamming_weight", key=int(k), **stats(s.i_tot.values, s.ml_T.values)))
    for k, s in D.groupby("parity"): rows.append(dict(group="parity_output", key=int(k), **stats(s.i_tot.values, s.ml_T.values)))
    D["Tbin"] = pd.cut(D.temp, [-41, -10, 20, 50, 80, 126]); D["Vbin"] = pd.cut(D.vdd, [.89, .94, .98, 1.02, 1.06, 1.11])
    for k, s in D.groupby("Tbin", observed=True): rows.append(dict(group="temperature_C", key=str(k), **stats(s.i_tot.values, s.ml_T.values)))
    for k, s in D.groupby("Vbin", observed=True): rows.append(dict(group="VDD_V", key=str(k), **stats(s.i_tot.values, s.ml_T.values)))
    pd.DataFrame(rows).assign(quantity="T_total_VDD_level_current").to_csv(os.path.join(CD, "circuit_results_table_T.csv"), index=False)
    rowsS = [dict(group="ALL", key="all", **stats(D.i_vdd.values, D.ml_S.values))] + [dict(group="hamming_weight", key=int(k), **stats(s.i_vdd.values, s.ml_S.values)) for k, s in D.groupby("hamming")]
    rowsS.append(dict(group="ALL", key="all_load_corrected", **stats(D.i_vdd.values, D.ml_S_loadcorr.values)))
    pd.DataFrame(rowsS).assign(quantity=lambda d: np.where(d.key == "all_load_corrected", "S_VDD_rail_current_only_load_corrected", "S_VDD_rail_current_only_plain_gate_sum")).to_csv(os.path.join(CD, "circuit_results_table_S.csv"), index=False)
    D["err_T"] = (D.ml_T - D.i_tot) / D.i_tot
    D.reindex(D.err_T.abs().sort_values(ascending=False).index).head(100)[["corner", "vec", "hamming", "temp", "vdd", "i_tot", "ml_T", "err_T"]].to_csv(os.path.join(CD, "circuit_worst_records.csv"), index=False)
    for q, (ref, ml, fn) in {"T": ("i_tot", "ml_T", "circuit_parity_T.png"), "S": ("i_vdd", "ml_S", "circuit_parity_S.png")}.items():
        fig, ax = plt.subplots(figsize=(5.6, 5.2)); sc = ax.scatter(D[ref], D[ml], c=D.temp, s=1.5, alpha=.35, cmap="viridis"); ax.set_xscale("log"); ax.set_yscale("log")
        lo, hi = D[ref].min() / 1.5, D[ref].max() * 1.5; ax.plot([lo, hi], [lo, hi], "k--", lw=.8); plt.colorbar(sc, label="T (C)")
        ax.set_xlabel("SPICE complete-circuit leakage (A)"); ax.set_ylabel("Hierarchical ML leakage (A)"); ax.set_title(f"8-bit parity checker, {len(D):,} points ({q})"); ax.grid(alpha=.3, which="both")
        plt.tight_layout(); plt.savefig(os.path.join(CD, fn), dpi=140); plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(D.temp, 100 * D.err_T, ".", ms=1, alpha=.3); ax[0].set_xlabel("T (C)"); ax[0].set_ylabel("signed error (%)")
    ax[1].plot(D.vdd, 100 * D.err_T, ".", ms=1, alpha=.3); ax[1].set_xlabel("VDD (V)")
    ax[2].hist(100 * D.err_T, bins=100); ax[2].set_xlabel("signed error (%)")
    for a in ax: a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(CD, "circuit_error_analysis.png"), dpi=130); plt.close()
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in M.items()}, indent=1))
