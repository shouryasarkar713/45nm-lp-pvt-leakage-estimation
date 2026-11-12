"""Central configuration. Portable: everything is relative to DVD_ROOT (default: parent of src/)."""
import os, re, hashlib
ROOT = os.environ.get("DVD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_CARD = os.path.join(ROOT, "model", "45nm_LP.pm")
WORK = os.environ.get("DVD_WORK", os.path.join(ROOT, "work"))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data")
for d in (WORK, OUT, DATA, os.path.join(OUT, "audit"), os.path.join(OUT, "tables"), os.path.join(OUT, "plots")):
    os.makedirs(d, exist_ok=True)

# ---- assignment ----
TECH = "45nm_LP"; LMIN = 45e-9; VDD_NOM = 1.0
VDD_LO, VDD_HI = 0.9 * VDD_NOM, 1.1 * VDD_NOM
T_LO, T_HI = -40.0, 125.0
W_LO, W_HI = 1 * LMIN, 8 * LMIN
PROC = ["vth0", "toxe", "ndep", "u0", "xj"]
CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
CASE_DEF = {  # device, gate, body (symbolic: 'VDD' or 0)
    "NMOS_ON": ("nmos", "VDD", 0), "NMOS_OFF": ("nmos", 0, 0),
    "PMOS_ON": ("pmos", 0, "VDD"), "PMOS_OFF": ("pmos", "VDD", "VDD")}
OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"
WN, WP = 2 * LMIN, 4 * LMIN         # gate/circuit device widths (inside 1..8 Lmin)
N_SPEC = 10000
SEEDS = dict(spec=[101, 102, 103, 104], split=2024, corners=303, ml=404, extra=500)
N_CORNERS = 200
NGSPICE_ENV = "NGSPICE"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def parse_card(path=MODEL_CARD):
    """Return (full text, {'nmos':{p:v}, 'pmos':{p:v}}) for the swept params."""
    txt = open(path).read()
    parts = re.split(r"(?im)^\.model\s+", txt)
    nom = {}
    for p in parts[1:]:
        t = p.split()[1].lower()
        nom[t] = {k: float(re.search(r"(?<![a-z0-9_])" + k + r"\s*=\s*([-+0-9.eE]+)", p).group(1)) for k in PROC}
    return txt, nom
CARD_TEXT, NOMINAL = parse_card()

def card_with(nmos=None, pmos=None):
    """Model-card text with the swept parameters of nmos/pmos replaced (same override as `altermod`)."""
    out = []
    cur = None
    for line in CARD_TEXT.splitlines():
        m = re.match(r"(?i)^\.model\s+\S+\s+(\S+)", line)
        if m:
            cur = m.group(1).lower()
        ov = {"nmos": nmos, "pmos": pmos}.get(cur)
        if ov:
            for k, v in ov.items():
                line = re.sub(r"(?<![a-zA-Z0-9_])" + k + r"(\s*=\s*)[-+0-9.eE]+", lambda mm: f"{k}{mm.group(1)}{v:.10g}", line)
        out.append(line)
    return "\n".join(out) + "\n"
