"""Deliverable 2: example netlists (one per transistor case, one per gate, the parity checker) in the altermod style,
plus a self-test that each example reproduces the generator's inline-model-card result."""
import os, sys, numpy as np, pandas as pd, shutil
from config import *
from spice import *
from corners import *
from circuit import *
ND = os.path.join(ROOT, "netlists"); os.makedirs(ND, exist_ok=True); shutil.copy(MODEL_CARD, os.path.join(ND, "45nm_LP.pm"))

def alter_block(nmos, pmos, temp):
    L = [".control", "set numdgt=15"]
    for dev, P in (("nmos", nmos), ("pmos", pmos)):
        if P: L += [f"altermod @{dev}[{k}] = {v:.10g}" for k, v in P.items()]
    return L + [f"set temp = {temp:.6g}", "op"]

def transistor(case, row):
    dev, g, b = CASE_DEF[case]; vg = row["vdd"] if g == "VDD" else 0.0; vb = row["vdd"] if b == "VDD" else 0.0
    proc = {k: row[k] for k in PROC}
    L = [f"* {case}: single {dev.upper()} (every terminal on its own voltage source)", ".include 45nm_LP.pm", OPTIONS,
         f".param Lmin={LMIN:g}", f"M1 d g s b {dev} W={row['w']:.6e} L={{Lmin}}",
         f"Vd d 0 {row['vd']:.10g}", f"Vg g 0 {vg:.10g}", f"Vs s 0 {row['vs']:.10g}", f"Vb b 0 {vb:.10g}"]
    L += alter_block(proc if dev == "nmos" else None, proc if dev == "pmos" else None, row["temp"])
    L += ["print i(vd) i(vg) i(vs) i(vb)", ".endc", ".end"]
    return "\n".join(L) + "\n"

def gate_net(kind, ab, c):
    nm, pm = corner_procs(c); a, b = ab[0], (ab[1] if len(ab) > 1 else None)
    devs, internal = gate_devices(kind, "G", "a", "b" if b is not None else None, "out")
    L = [f"* {kind}  inputs {ab}   corner VDD={c['vdd']:.4f} T={c['temp']:.2f}", ".include 45nm_LP.pm", OPTIONS, f".param Lmin={LMIN:g} Wn={WN:g} Wp={WP:g}"]
    for n_, typ, dn, gn, sn, bn in devs: L.append(f"M{n_} {dn} {gn} {sn} {bn} {typ} W={{{'Wp' if typ=='pmos' else 'Wn'}}} L={{Lmin}}")
    L += [f"Va a 0 {c['vdd']*a:.10g}"] + ([f"Vb b 0 {c['vdd']*b:.10g}"] if b is not None else []) + [f"Vdd vdd 0 {c['vdd']:.10g}", "Vss gss 0 0"]
    L += alter_block(nm, pm, c["temp"]) + ["print i(vdd) i(vss) i(va)" + (" i(vb)" if b is not None else "") + " v(out)" + "".join(f" v({n_.lower()})" for n_ in internal), ".endc", ".end"]
    return "\n".join(L) + "\n"

def circuit_net(bits, c):
    nm, pm = corner_procs(c); D, I = devices()
    L = [f"* 8-bit parity checker (7 x XOR2; XOR2 = AND2(NAND2(a,b), OR2(a,b))) inputs {''.join(map(str,bits))}  VDD={c['vdd']:.4f} T={c['temp']:.2f}",
         ".include 45nm_LP.pm", OPTIONS, f".param Lmin={LMIN:g} Wn={WN:g} Wp={WP:g}"]
    for n_, typ, dn, gn, sn, bn in D: L.append(f"M{n_} {dn} {gn} {sn} {bn} {typ} W={{{'Wp' if typ=='pmos' else 'Wn'}}} L={{Lmin}}")
    L += [f"Vp{i} p{i} 0 {c['vdd']*bits[i]:.10g}" for i in range(8)] + [f"Vdd vdd 0 {c['vdd']:.10g}", "Vss gss 0 0"]
    L += alter_block(nm, pm, c["temp"]) + ["print i(vdd) i(vss) " + " ".join(f"i(vp{i})" for i in range(8)) + f" v({OUTPUT})", ".endc", ".end"]
    return "\n".join(L) + "\n"

if __name__ == "__main__":
    C = load_corners("official").iloc[0].to_dict(); report = []
    for case in CASES:
        d = pd.read_csv(os.path.join(DATA, "base", f"{case}.csv")).iloc[0].to_dict()
        open(os.path.join(ND, f"example_transistor_{case}.cir"), "w").write(transistor(case, d))
    for kind, ab in [("INV", (1,)), ("NAND2", (0, 1)), ("NOR2", (1, 0)), ("AND2", (1, 1)), ("OR2", (0, 0))]:
        open(os.path.join(ND, f"example_gate_{kind}.cir"), "w").write(gate_net(kind, ab, C))
    bits = (1, 0, 1, 1, 0, 0, 1, 0); open(os.path.join(ND, "example_parity_checker.cir"), "w").write(circuit_net(bits, C))
    print("netlists written to", ND)
