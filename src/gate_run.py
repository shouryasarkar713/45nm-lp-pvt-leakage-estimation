"""Deliverables 6,7: gate-level ML vs SPICE on the OFFICIAL corners (evaluated once, after the models are frozen)."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import *
from corners import load_corners
from gate_eval import predict_gates
from gate_nodes import solve_nodes
from train_models import load_chosen
from ml_data import features

GD = os.path.join(OUT, "gate"); os.makedirs(GD, exist_ok=True)

def make_provider():
    M = {c: load_chosen(c) for c in CASES}
    def provider(case, df):
        m, kind = M[case]
        return m.predict(df) if kind == "StructMLP" else m.predict(features(df, case))
    return provider

def stats(ref, ml):
    e = (ml - ref) / ref
    return dict(n=len(ref), spice_median_A=np.median(ref), ml_median_A=np.median(ml), bias_pct=100 * np.median(e),
                median_err_pct=100 * np.median(np.abs(e)), p95_err_pct=100 * np.quantile(np.abs(e), .95),
                mean_err_pct=100 * np.mean(np.abs(e)), mean_accuracy_pct=100 * (1 - np.mean(np.abs(e))))

def table(g, ref, ml):
    rows = []
    for (kind, a, b), s in g.groupby(["kind", "a", "b"], sort=False):
        rows.append(dict(gate=kind, a=a, b=("" if b < 0 else b), **stats(s[ref].values, s[ml].values)))
    return pd.DataFrame(rows)

def parity(g, ref, ml, title, fn):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for kind, s in g.groupby("kind"): ax.loglog(s[ref], s[ml], ".", ms=3, alpha=.5, label=kind)
    lo, hi = g[ref].min() / 2, g[ref].max() * 2; ax.plot([lo, hi], [lo, hi], "k--", lw=.8, label="y=x")
    ax.set_xlabel("SPICE leakage (A)"); ax.set_ylabel("ML leakage (A)"); ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
    plt.tight_layout(); plt.savefig(fn, dpi=140); plt.close()

if __name__ == "__main__":
    g = pd.read_csv(os.path.join(DATA, "gate_spice_official.csv")); assert g.ok.all() and len(g) == 3600
    C = load_corners("official"); prov = make_provider(); gg, dr = predict_gates(g, C, prov)
    # supplementary: ML-only stack-node voltage (KCL root-finding with the transistor models) vs SPICE stack node
    gg["ml_stack_node"] = solve_nodes(gg, dr, prov)
    gg["spice_stack_node"] = np.select([gg.kind == "NAND2", gg.kind == "NOR2", gg.kind == "AND2", gg.kind == "OR2"], [gg.get("v_x"), gg.get("v_y"), gg.get("v_G1_x"), gg.get("v_G1_y")], default=np.nan)
    nd = gg[gg.spice_stack_node.notna()].copy(); nd["abs_err_V"] = (nd.ml_stack_node - nd.spice_stack_node).abs(); nd["dlog10"] = np.abs(np.log10(nd.ml_stack_node.clip(1e-12)) - np.log10(nd.spice_stack_node.clip(1e-12)))
    nd.groupby(["kind", "a", "b"]).agg(n=("abs_err_V", "size"), median_abs_err_V=("abs_err_V", "median"), p95_abs_err_V=("abs_err_V", lambda x: x.quantile(.95)), median_dlog10=("dlog10", "median")).reset_index().to_csv(os.path.join(GD, "gate_stack_node_check.csv"), index=False)
    gg.drop(columns=["_idx"]).to_csv(os.path.join(GD, "gate_results_per_point.csv"), index=False); dr.to_csv(os.path.join(GD, "gate_device_predictions.csv"), index=False)
    tT = table(gg, "i_tot_vdd", "ml_T"); tS = table(gg, "i_vdd", "ml_S")
    tT.assign(quantity="T_total_VDD_level_current").to_csv(os.path.join(GD, "gate_results_table_T.csv"), index=False); tS.assign(quantity="S_VDD_rail_current_only").to_csv(os.path.join(GD, "gate_results_table_S.csv"), index=False)
    allT = stats(gg.i_tot_vdd.values, gg.ml_T.values); allS = stats(gg.i_vdd.values, gg.ml_S.values)
    json.dump(dict(T=allT, S=allS), open(os.path.join(GD, "gate_overall_metrics.json"), "w"), indent=1, default=float)
    parity(gg, "i_tot_vdd", "ml_T", "Gate leakage T (all 3,600 gate simulations)", os.path.join(GD, "gate_parity_T.png"))
    parity(gg, "i_vdd", "ml_S", "Gate leakage S = I(VDD rail)", os.path.join(GD, "gate_parity_S.png"))
    # robustness breakdowns (PVT) and worst cases
    gg = gg.merge(C[["corner", "nmos_vth0", "pmos_vth0", "nmos_toxe", "pmos_toxe"]], on="corner")
    gg["e"] = (gg.ml_T - gg.i_tot_vdd) / gg.i_tot_vdd
    rows = []
    for name, col, bins in (("temperature_C", "temp", [-41, -10, 20, 50, 80, 126]), ("VDD_V", "vdd", [.89, .94, .98, 1.02, 1.06, 1.11])):
        for k, s in gg.groupby(pd.cut(gg[col], bins), observed=True): rows.append(dict(group=name, key=str(k), **stats(s.i_tot_vdd.values, s.ml_T.values)))
    for name, col in (("nmos_vth0", "nmos_vth0"), ("pmos_|vth0|", "pmos_vth0"), ("nmos_toxe", "nmos_toxe"), ("pmos_toxe", "pmos_toxe")):
        x = gg[col].abs(); q = pd.qcut(x, 4, duplicates="drop")
        for k, s in gg.groupby(q, observed=True): rows.append(dict(group=name + "_quartile", key=str(k), **stats(s.i_tot_vdd.values, s.ml_T.values)))
    for k, s in gg.groupby("kind"): rows.append(dict(group="gate", key=k, **stats(s.i_tot_vdd.values, s.ml_T.values)))
    pd.DataFrame(rows).to_csv(os.path.join(GD, "gate_robustness_T.csv"), index=False)
    gg.reindex(gg.e.abs().sort_values(ascending=False).index).head(60)[["kind", "a", "b", "corner", "temp", "vdd", "nmos_vth0", "pmos_vth0", "i_tot_vdd", "ml_T", "e"]].to_csv(os.path.join(GD, "gate_worst_records.csv"), index=False)
    print(tT.round(3).to_string()); print("OVERALL T", allT); print("OVERALL S", allS)
