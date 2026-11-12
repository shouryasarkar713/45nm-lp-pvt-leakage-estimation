"""ngspice runner + netlist builders + parsers (one self-contained process per operating point)."""
import os, re, subprocess, shutil, tempfile, itertools
import numpy as np
from config import *

def ngspice_bin():
    p = os.environ.get(NGSPICE_ENV) or shutil.which("ngspice")
    if not p: raise RuntimeError("ngspice not found: put it on PATH or set $NGSPICE (optionally $NGSPICE_LD_LIBRARY_PATH)")
    return p

_TMP = None
def tmpdir():
    global _TMP
    if _TMP is None:
        base = "/dev/shm" if os.path.isdir("/dev/shm") else None
        _TMP = tempfile.mkdtemp(prefix="dvd_", dir=base)
    return _TMP

def run_netlist(text, timeout=300):
    env = dict(os.environ)
    ld = os.environ.get("NGSPICE_LD_LIBRARY_PATH")
    if ld: env["LD_LIBRARY_PATH"] = ld + ":" + env.get("LD_LIBRARY_PATH", "")
    env["OMP_NUM_THREADS"] = "1"
    path = os.path.join(tmpdir(), f"n{os.getpid()}.cir")
    with open(path, "w") as f: f.write(text)
    pr = subprocess.run([ngspice_bin(), "-b", path], capture_output=True, text=True, env=env, timeout=timeout)
    return pr.stdout + "\n" + pr.stderr

_NUM = r"([-+]?[0-9.]+(?:[eE][-+]?\d+)?|[-+]?nan|[-+]?inf)"
def parse_prints(out):
    """All 'name = value' lines from the .control print section (last occurrence wins)."""
    d = {}
    for m in re.finditer(r"^(\S+)\s*=\s*" + _NUM + r"\s*$", out, flags=re.M):
        d[m.group(1).lower()] = float(m.group(2))
    return d

# Deterministic, ML-independent solver ladder.  Level 0 is the assignment's option set; later levels relax ONLY the
# convergence tolerances (strict tolerances are numerically unreachable for some BSIM4 operating points, see report).
LADDER = [
    ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9",
    ".options gmin=1e-18 abstol=1e-16 reltol=1e-6 vntol=1e-9",
    ".options gmin=1e-18 abstol=1e-15 reltol=1e-6 vntol=1e-9",
    ".options gmin=1e-18 abstol=1e-15 reltol=1e-4 vntol=1e-9",
    ".options gmin=1e-18 abstol=1e-15 reltol=1e-3 vntol=1e-9",
    ".options gmin=1e-18 abstol=1e-15 reltol=1e-3 vntol=1e-9 gminsteps=100 srcsteps=100 itl1=500",
]
HEAD = "* DVD-ML 45nm LP leakage project (auto-generated)\n"
def control(prints, temp):
    return (f".temp {temp:.6g}\n.control\nset numdgt=15\nop\n" +
            "".join(f"print {p}\n" for p in prints) + ".endc\n.end\n")

# ---------------------------------------------------------------- single transistor
def transistor_netlist(case, vdd, temp, w, vd, vs, proc, model_include=None):
    dev, g, b = CASE_DEF[case]
    vg = vdd if g == "VDD" else 0.0; vb = vdd if b == "VDD" else 0.0
    card = card_with(**{dev: proc})
    nl = (HEAD + card + OPTIONS + "\n" +
          f"M1 d g s b {dev} W={w:.6e} L={LMIN:.6e}\n"
          f"Vd d 0 {vd:.10g}\nVg g 0 {vg:.10g}\nVs s 0 {vs:.10g}\nVb b 0 {vb:.10g}\n")
    nl += control(["i(vd)", "i(vg)", "i(vs)", "i(vb)"], temp)
    return nl, vg, vb

def sim_device(typ, temp, w, vd, vg, vs, vb, proc):
    """Isolated single device, every terminal on its own ideal source -> currents INTO d,g,s,b."""
    nl = (HEAD + card_with(**{typ: proc}) + OPTIONS + "\n" +
          f"M1 d g s b {typ} W={w:.6e} L={LMIN:.6e}\n"
          f"Vd d 0 {vd:.15g}\nVg g 0 {vg:.15g}\nVs s 0 {vs:.15g}\nVb b 0 {vb:.15g}\n" +
          control(["i(vd)", "i(vg)", "i(vs)", "i(vb)"], temp))
    out = run_netlist(nl); p = parse_prints(out)
    if "failed" in out.lower() or any(k not in p for k in ("i(vd)", "i(vg)", "i(vs)", "i(vb)")): return (np.nan,) * 4
    return (-p["i(vd)"], -p["i(vg)"], -p["i(vs)"], -p["i(vb)"])

def sim_transistor(args):
    case, vdd, temp, w, vd, vs, proc = args
    nl, vg, vb = transistor_netlist(case, vdd, temp, w, vd, vs, proc)
    p = parse_prints(run_netlist(nl))
    try:  # terminal current INTO the device = -i(source)
        return (-p["i(vd)"], -p["i(vg)"], -p["i(vs)"], -p["i(vb)"])
    except KeyError:
        return (np.nan,) * 4

# ---------------------------------------------------------------- gates / circuit
# Transistor list: (name, type, drain, gate, source, body); nets: vdd, gss (=0 V via Vss), inputs
def gate_devices(kind, pre, a, b, out):
    """Return device list + internal-node names for a gate instance. 'pre' makes names unique."""
    n = lambda s: f"{pre}_{s}"
    D = []; internal = []
    if kind == "INV":
        D = [(n("Mp"), "pmos", out, a, "vdd", "vdd"), (n("Mn"), "nmos", out, a, "gss", "gss")]
    elif kind == "NAND2":
        x = n("x"); internal = [x]
        D = [(n("MpA"), "pmos", out, a, "vdd", "vdd"), (n("MpB"), "pmos", out, b, "vdd", "vdd"),
             (n("MnA"), "nmos", out, a, x, "gss"), (n("MnB"), "nmos", x, b, "gss", "gss")]
    elif kind == "NOR2":
        y = n("y"); internal = [y]
        D = [(n("MpA"), "pmos", y, a, "vdd", "vdd"), (n("MpB"), "pmos", out, b, y, "vdd"),
             (n("MnA"), "nmos", out, a, "gss", "gss"), (n("MnB"), "nmos", out, b, "gss", "gss")]
    elif kind in ("AND2", "OR2"):
        mid = n("m"); first = "NAND2" if kind == "AND2" else "NOR2"
        d1, i1 = gate_devices(first, pre + "1", a, b, mid)
        d2, _ = gate_devices("INV", pre + "2", mid, None, out)
        D = d1 + d2; internal = i1 + [mid]
    else: raise ValueError(kind)
    return D, internal

def dev_line(d, wn=WN, wp=WP):
    name, typ, dn, gn, sn, bn = d
    w = wp if typ == "pmos" else wn
    return f"M{name} {dn} {gn} {sn} {bn} {typ} W={w:.6e} L={LMIN:.6e}"

def build_netlist(devices, vdd, temp, nmos, pmos, inputs, nodes, tag, amm=False):
    """inputs: {name: 0/1}.  amm=True inserts a 0 V ammeter in series with every device terminal so that all
    device terminal currents (id,ig,is,ib INTO the device = -i(ammeter)) are SPICE-native quantities."""
    L = [HEAD + f"* {tag}", card_with(nmos=nmos, pmos=pmos), OPTIONS]
    pr = ["i(vdd)", "i(vss)"] + [f"i(v{k})" for k in inputs] + [f"v({n})" for n in nodes]
    for d in devices:
        name, typ, dn, gn, sn, bn = d
        if amm:
            nets = [f"{name}_{t}" for t in "dgsb"]
            L.append(dev_line((name, typ, *nets)))
            for t, n_, net in zip("dgsb", (dn, gn, sn, bn), nets):
                L.append(f"Vm_{name}_{t} {net} {n_} 0"); pr.append(f"i(vm_{name.lower()}_{t})")
        else:
            L.append(dev_line(d))
    for k, v in inputs.items(): L.append(f"V{k} {k} 0 {vdd * v:.10g}")
    L += [f"Vdd vdd 0 {vdd:.10g}", "Vss gss 0 0"]
    L.append(control(pr, temp))
    return "\n".join(L)

def dev_currents(name, p):
    """(id, ig, is, ib) into the device from an ammeter-instrumented run."""
    return tuple(-p.get(f"i(vm_{name.lower()}_{t})", np.nan) for t in "dgsb")

def gate_case(kind):
    return {"INV": [(0,), (1,)]}.get(kind, [(0, 0), (0, 1), (1, 0), (1, 1)])

def sim_gate(args):
    kind, ab, vdd, temp, nmos, pmos = args
    a, b = ab[0], (ab[1] if len(ab) > 1 else None)
    devs, internal = gate_devices(kind, "G", "a", "b" if b is not None else None, "out")
    inputs = {"a": a}
    if b is not None: inputs["b"] = b
    nodes = ["out"] + internal
    nl = build_netlist(devs, vdd, temp, nmos, pmos, inputs, nodes, f"{kind} {ab}")
    p = parse_prints(run_netlist(nl))
    return devs, inputs, nodes, p

def oracle_batch(devs_volts, nmos, pmos, temp, wn=WN, wp=WP):
    """Independent SPICE evaluation of many isolated devices (each terminal on its own ideal source) at given terminal
    voltages.  devs_volts: list of (name, typ, (vd,vg,vs,vb)).  Returns {name: (id,ig,is,ib)} INTO the devices."""
    L = [HEAD + card_with(nmos=nmos, pmos=pmos), OPTIONS]; pr = []
    for k, (name, typ, v) in enumerate(devs_volts):
        w = wp if typ == "pmos" else wn
        L.append(f"M{k} d{k} g{k} s{k} b{k} {typ} W={w:.6e} L={LMIN:.6e}")
        for t, x in zip("dgsb", v):
            L.append(f"V{t}{k} {t}{k} 0 {x:.15g}"); pr.append(f"i(v{t}{k})")
    L.append(control(pr, temp))
    out = run_netlist("\n".join(L)); p = parse_prints(out)
    res = {}
    for k, (name, typ, v) in enumerate(devs_volts):
        res[name] = tuple(-p.get(f"i(v{t}{k})", np.nan) for t in "dgsb")
    return res
