"""Part C.7/8 - hierarchical prediction of the 8-bit parity checker from PART-B GATE MODEL PREDICTIONS ONLY.
The prediction function receives: the 8 input bits, the corner id and the table of Part-B gate predictions.
It never touches the complete-circuit SPICE records (those are used only afterwards as the evaluation reference)."""
import itertools, os, glob
import numpy as np, pandas as pd
from config import *
from circuit import GATES, logic

VECS = list(itertools.product([0, 1], repeat=8))

def gate_instance_keys(bits):
    """Logic value at the inputs of every gate instance -> (kind, a, b) keys (INPUTS ARE EXACT LOGIC VALUES)."""
    v = logic(bits)
    return [(kind, v[a], v[b]) for kind, name, a, b, out in GATES]

def hierarchical_predict(gate_table, corners):
    """gate_table: dict {(kind,a,b,corner): value} of Part-B ML gate leakages (one quantity).  Returns DataFrame
    (corner, vec, ml) with the sum over the 21 gate instances for every input vector at every corner."""
    keys = [gate_instance_keys(b) for b in VECS]; rows = []
    for c in corners:
        for vi, ks in enumerate(keys):
            rows.append((c, vi, sum(gate_table[(k, a, b, c)] for k, a, b in ks)))
    return pd.DataFrame(rows, columns=["corner", "vec", "ml"])

def load_circuit_spice():
    fs = sorted(glob.glob(os.path.join(WORK, "circuit_spice", "corner_*.csv")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)

def hierarchical_predict_S_loadcorrected(tS, tIa, tIb, corners):
    """Rail-only quantity S of the whole circuit from Part-B gate predictions only.  In the circuit the gate current of a load whose
    input is an INTERNAL net held high is supplied through the driver from the VDD rail (it is in S); in the isolated gate it is supplied
    by the input source (not in S).  Hence S_circuit = sum_g [ S_g + 1(a internal & high) in_a,g + 1(b internal & high) in_b,g ]."""
    rows = []
    for c in corners:
        for vi, bits in enumerate(VECS):
            v = logic(bits); tot = 0.0
            for kind, name, a, b, out in GATES:
                k = (kind, v[a], v[b], c); tot += tS[k]
                if not a.startswith("p") and v[a]: tot += tIa[k]
                if not b.startswith("p") and v[b]: tot += tIb[k]
            rows.append((c, vi, tot))
    return pd.DataFrame(rows, columns=["corner", "vec", "ml"])
