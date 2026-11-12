"""Gate-level leakage from per-device terminal currents (Part B.4) - provider-agnostic."""
import numpy as np
from config import *
from spice import gate_devices

def net_voltages(kind, rec, devs):
    """Terminal-net voltages of a gate record (row dict-like) using SPICE-recorded node voltages + 0/VDD inputs."""
    vdd = rec["vdd"]; v = {"vdd": vdd, "gss": 0.0, "a": vdd * rec["a"], "out": rec["v_out"]}
    if rec["b"] >= 0: v["b"] = vdd * rec["b"]
    for k in ("v_G1_x", "v_x"): pass
    for key in ("x", "y", "m"):
        if f"v_{key}" in rec and rec[f"v_{key}"] == rec[f"v_{key}"]: v[f"G_{key}"] = rec[f"v_{key}"]
    if "v_G1_x" in rec and rec["v_G1_x"] == rec["v_G1_x"]: v["G1_x"] = rec["v_G1_x"]
    if "v_G1_y" in rec and rec["v_G1_y"] == rec["v_G1_y"]: v["G1_y"] = rec["v_G1_y"]
    return v

def gate_structure(kind):
    devs, internal = gate_devices(kind, "G", "a", "b" if kind != "INV" else None, "out")
    return devs, internal

def supply_from_currents(devs, cur, high_nets):
    """cur: {devname: (id,ig,is,ib)} into the device.  Returns (S, T): current delivered by the VDD rail only (S) and by
    all VDD-level sources = rail + inputs held at VDD (T)."""
    S = 0.0; T = 0.0
    for name, typ, dn, gn, sn, bn in devs:
        c = cur[name]
        for node, i in zip((dn, gn, sn, bn), c):
            if node == "vdd": S += i; T += i
            elif node in high_nets: T += i
    return S, T
