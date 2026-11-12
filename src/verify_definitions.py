"""Independent verification of the leakage quantities T and S from raw voltage-source currents (no ML involved)."""
import json, os, glob, itertools, numpy as np, pandas as pd
from config import *
from circuit_eval import load_circuit_spice
if __name__ == "__main__":
    out = {}
    g = pd.read_csv(os.path.join(DATA, "gate_spice_official.csv"))
    P = g.vdd * g.i_vdd + g.vdd * (g.a * g.i_in_a) + g.vdd * (g.b.clip(lower=0) * g.i_in_b)   # sum_k V_k * I_k delivered (inputs at 0 V deliver no power)
    out["gate_T_equals_static_power_over_VDD_max_rel_diff"] = float((P / g.vdd / g.i_tot_vdd - 1).abs().max())
    kcl = g.i_vdd + g.i_in_a + g.i_in_b.where(g.b >= 0, 0) - g.i_vss      # currents delivered by vdd + inputs = current absorbed by Vss (sign: i_vss is into the 0 V source)
    out["gate_source_current_balance_max_rel"] = float((kcl.abs() / g.i_tot_vdd).max())
    out["gate_S_over_T_median"] = float((g.i_vdd / g.i_tot_vdd).median())
    out["gate_cases_with_S_ne_T_gt_1pct"] = float(((g.i_tot_vdd - g.i_vdd).abs() / g.i_tot_vdd > .01).mean())
    C = load_circuit_spice(); bits = np.array(list(itertools.product([0, 1], repeat=8)))[C.vec.values]
    Cc = pd.read_csv(os.path.join(OUT, "circuit", "circuit_results_per_point.csv.gz")); vdd = Cc.vdd.values
    Pc = vdd * C.i_vdd.values + vdd * sum(bits[:, k] * C[f"i_p{k}"].values for k in range(8))
    out["circuit_T_equals_static_power_over_VDD_max_rel_diff"] = float(np.abs(Pc / vdd / C.i_tot.values - 1).max())
    bal = C.i_vdd.values + sum(C[f"i_p{k}"].values for k in range(8)) - C.i_vss.values
    out["circuit_source_balance_max_rel"] = float(np.abs(bal / C.i_tot.values).max())
    out["circuit_S_over_T_median"] = float((C.i_vdd / C.i_tot).median())
    # load-correction identity checked on SPICE data only (gate currents drawn by internal high nets are supplied by the VDD rail)
    out["load_correction_SPICE_only_mean_abs_err_pct"] = float(100 * ((Cc.gsum_S_loadcorr - Cc.i_vdd) / Cc.i_vdd).abs().mean())
    out["plain_gate_sum_S_mean_abs_err_pct"] = float(100 * ((Cc.gsum_S - Cc.i_vdd) / Cc.i_vdd).abs().mean())
    out["T_gate_sum_mean_abs_err_pct"] = float(100 * ((Cc.gsum_T - Cc.i_tot) / Cc.i_tot).abs().mean())
    json.dump(out, open(os.path.join(OUT, "audit", "definition_verification.json"), "w"), indent=1); print(json.dumps(out, indent=1))
