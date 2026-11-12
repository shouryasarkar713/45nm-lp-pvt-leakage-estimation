"""8-bit parity checker: 7 x XOR2, each XOR2 = AND2(NAND2(a,b), OR2(a,b)) -> 21 gates, 112 transistors."""
from config import *
from spice import gate_devices

def topology():
    """Return list of gate instances (kind, name, in_a, in_b, out) in topological order + XOR tree structure."""
    gates = []; level = [f"p{i}" for i in range(8)]; k = 0
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            k += 1; a, b = level[i], level[i + 1]
            gates += [("NAND2", f"N{k}", a, b, f"n{k}"), ("OR2", f"O{k}", a, b, f"o{k}"), ("AND2", f"A{k}", f"n{k}", f"o{k}", f"x{k}")]
            nxt.append(f"x{k}")
        level = nxt
    return gates, level[0]

GATES, OUTPUT = topology()

def devices():
    D = []; internal = []
    for kind, name, a, b, out in GATES:
        d, i = gate_devices(kind, name, a, b, out); D += d; internal += i
    return D, internal

def logic(bits):
    """bits: 8 ints -> dict net -> value (Boolean simulation of the netlist)."""
    v = {f"p{i}": int(bits[i]) for i in range(8)}
    f = {"NAND2": lambda a, b: 1 - (a & b), "OR2": lambda a, b: a | b, "AND2": lambda a, b: a & b}
    for kind, name, a, b, out in GATES: v[out] = f[kind](v[a], v[b])
    return v

if __name__ == "__main__":
    D, I = devices(); print(len(GATES), "gates", len(D), "devices", len(I), "internal", OUTPUT)
    import itertools
    ok = all(logic(b)[OUTPUT] == sum(b) % 2 for b in itertools.product([0, 1], repeat=8))
    print("truth table = parity of 8 bits:", ok)
