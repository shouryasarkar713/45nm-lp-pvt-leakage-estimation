"""Per-input-vector circuit results (256 vectors x 200 corners) aggregated from the EXISTING circuit results file - no new simulation.
Definitions follow circuit_run.stats: relative error e = (ML - SPICE)/SPICE; bias_pct = 100 x MEDIAN of the signed relative error (same convention as all gate/circuit tables);
mean_signed_err_pct is also given; median_error_pct / p95_error_pct are on |e|; mean_mape_pct = 100 x mean|e|; mean_accuracy_pct = 100 - mean_mape_pct."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats as st
from config import *
CD = os.path.join(OUT, "circuit")
QUANT = {"T": ("i_tot", "ml_T", "T_total_VDD_level_current"), "S": ("i_vdd", "ml_S", "S_VDD_rail_current_only_plain_gate_sum"), "S_loadcorrected": ("i_vdd", "ml_S_loadcorr", "S_VDD_rail_current_only_load_corrected")}

def per_vector(D, ref, ml, qname):
    rows = []
    for v, s in D.groupby("vec"):
        y = s[ref].values; p = s[ml].values; e = (p - y) / y
        rows.append(dict(vector=int(v), input_bits=format(int(v), "08b"), parity=int(bin(int(v)).count("1") % 2), hamming_weight=bin(int(v)).count("1"), pvt_count=len(s),
                         spice_median=np.median(y), ml_median=np.median(p), mean_spice=y.mean(), mean_ml=p.mean(), bias_pct=100 * np.median(e), mean_signed_err_pct=100 * e.mean(),
                         median_error_pct=100 * np.median(np.abs(e)), p95_error_pct=100 * np.quantile(np.abs(e), .95), max_error_pct=100 * np.abs(e).max(), mean_mape_pct=100 * np.abs(e).mean(),
                         mean_accuracy_pct=100 * (1 - np.abs(e).mean()), quantity=qname))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    D = pd.read_csv(os.path.join(CD, "circuit_results_per_point.csv.gz")); M = json.load(open(os.path.join(CD, "circuit_overall_metrics.json"))); summ = {}
    for q, (ref, ml, name) in QUANT.items():
        P = per_vector(D, ref, ml, name); fn = {"T": "circuit_results_per_vector_T.csv", "S": "circuit_results_per_vector_S.csv", "S_loadcorrected": "circuit_results_per_vector_S_loadcorrected.csv"}[q]
        P.to_csv(os.path.join(CD, fn), index=False)
        gk = {"T": "T", "S": "S", "S_loadcorrected": "S_load_corrected"}[q]
        summ[q] = dict(rows=len(P), global_mean_accuracy_from_vectors=float(100 - P.mean_mape_pct.mean()), global_mean_accuracy_reported=M[gk]["mean_accuracy_pct"],
                       worst_vector=int(P.loc[P.median_error_pct.idxmax(), "vector"]), best_vector=int(P.loc[P.median_error_pct.idxmin(), "vector"]))
    P = pd.read_csv(os.path.join(CD, "circuit_results_per_vector_T.csv")); a = {}
    a["spearman_hamming_vs_median_error"] = [float(x) for x in st.spearmanr(P.hamming_weight, P.median_error_pct)]
    a["spearman_hamming_vs_p95_error"] = [float(x) for x in st.spearmanr(P.hamming_weight, P.p95_error_pct)]
    a["mannwhitney_parity_median_error_p"] = float(st.mannwhitneyu(P[P.parity == 0].median_error_pct, P[P.parity == 1].median_error_pct).pvalue)
    a["median_of_median_error_by_parity"] = P.groupby("parity").median_error_pct.median().to_dict()
    a["median_error_range_pct"] = [float(P.median_error_pct.min()), float(P.median_error_pct.max())]; a["p95_error_range_pct"] = [float(P.p95_error_pct.min()), float(P.p95_error_pct.max())]
    a["top10_worst_median_error_hamming"] = P.nlargest(10, "median_error_pct").hamming_weight.tolist(); a["hamming_weight_counts_all"] = P.hamming_weight.value_counts().sort_index().to_dict()
    # a vector's records share the same 200 corners -> any difference between vectors reflects only the logic state, so the spread among vectors is compared with the corner spread
    a["spread_of_vector_median_error_over_mean"] = float(P.median_error_pct.std() / P.median_error_pct.mean())
    summ["association_analysis_T"] = a; json.dump(summ, open(os.path.join(CD, "circuit_per_vector_summary.json"), "w"), indent=1, default=str); print(json.dumps(summ, indent=1, default=str))
    for col, title, fn in (("median_error_pct", "median", "circuit_per_vector_median_error.png"), ("p95_error_pct", "95th-percentile", "circuit_per_vector_p95_error.png")):
        fig, ax = plt.subplots(figsize=(11, 3.6)); c = np.where(P.parity == 1, "#d95f02", "#1b9e77")
        ax.bar(P.vector, P[col], color=c, width=1.0); w = P.nlargest(5, col)
        for r in w.itertuples(): ax.annotate(f"{r.vector}\n({r.input_bits})", (r.vector, getattr(r, col)), fontsize=6, ha="center", va="bottom")
        ax.set_xlabel("input vector (decimal, p7..p0 = bits)"); ax.set_ylabel(f"{title} |relative error| (%)"); ax.set_title(f"Per-vector {title} error of T over 200 PVT corners (orange: odd parity, green: even; five worst labelled)")
        ax.set_xlim(-1, 256); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(CD, fn), dpi=140); plt.close()
    fig, ax = plt.subplots(figsize=(5.5, 3.6)); ax.boxplot([P[P.hamming_weight == h].median_error_pct for h in range(9)], tick_labels=range(9)); ax.set_xlabel("Hamming weight"); ax.set_ylabel("per-vector median error (%)"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(CD, "circuit_per_vector_by_hamming.png"), dpi=140); plt.close()
