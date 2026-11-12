"""Validation of the supply reconstruction: SPICE-native device currents (isolated single-device runs at the recorded terminal voltages,
stored by gen_gate.py) summed with the same rules as the ML prediction must reproduce the complete-gate SPICE leakage."""
import json, os, numpy as np, pandas as pd
from config import *
from gate_model import gate_structure, supply_from_currents
if __name__ == "__main__":
    out = {}
    for name in ("official", "dev"):
        d = pd.read_csv(os.path.join(DATA, f"gate_spice_{name}.csv")); d = d[d.ok == 1]; eS = []; eT = []
        for r in d.to_dict("records"):
            devs, _ = gate_structure(r["kind"]); cur = {dv[0]: tuple(r[f"{dv[0]}.{t}"] for t in ("id", "ig", "is", "ib")) for dv in devs}
            hi = {n for n, k in (("a", "a"), ("b", "b")) if r[k] == 1}; S, T = supply_from_currents(devs, cur, hi)
            eS.append(abs(S - r["i_vdd"]) / abs(r["i_vdd"])); eT.append(abs(T - r["i_tot_vdd"]) / abs(r["i_tot_vdd"]))
        eS, eT = np.array(eS), np.array(eT)
        out[name] = dict(n=len(d), S=dict(median=float(np.median(eS)), p99=float(np.quantile(eS, .99)), max=float(eS.max())), T=dict(median=float(np.median(eT)), p99=float(np.quantile(eT, .99)), max=float(eT.max())))
    os.makedirs(os.path.join(OUT, "audit"), exist_ok=True); json.dump(out, open(os.path.join(OUT, "audit", "oracle_reconstruction.json"), "w"), indent=1); print(json.dumps(out, indent=1))
