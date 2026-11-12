import sys, platform, subprocess, importlib, json, os
from config import *
from spice import ngspice_bin
info = dict(python=sys.version.split()[0], platform=platform.platform(), model_card_sha256=sha256(MODEL_CARD))
for m in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "torch", "matplotlib", "joblib", "docx"):
    try: info[m] = importlib.import_module(m).__version__
    except Exception as e: info[m] = "NOT INSTALLED"
env = dict(os.environ); ld = env.get("NGSPICE_LD_LIBRARY_PATH")
if ld: env["LD_LIBRARY_PATH"] = ld + ":" + env.get("LD_LIBRARY_PATH", "")
out = subprocess.run([ngspice_bin(), "--version"], capture_output=True, text=True, env=env).stdout
info["ngspice"] = [l for l in out.splitlines() if "ngspice-" in l][:1]
info["ngspice_path"] = "ngspice (resolved from PATH or $NGSPICE)"
os.makedirs(OUT, exist_ok=True); json.dump(info, open(os.path.join(OUT, "environment.json"), "w"), indent=1); print(json.dumps(info, indent=1))
