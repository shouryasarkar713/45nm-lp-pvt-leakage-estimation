"""Training-only 'zero-Vds twin': same sample but with vd := vs, giving the Vds=0 leakage offsets (id0, ig0, is0, ib0).
Used ONLY for rows that are in the training partition (see split.py); never for validation/test rows."""
import sys, os
import numpy as np, pandas as pd
from multiprocessing import Pool
from config import *
from spice import sim_transistor

def twin_file(path):
    case = os.path.basename(path).split("_uniform")[0].split("_gatecorner")[0].split("_nearzero")[0].replace(".csv", "")
    df = pd.read_csv(path)
    args = [(case, r.vdd, r.temp, r.w, r.vs, r.vs, {k: getattr(r, k) for k in PROC}) for r in df.itertuples()]
    with Pool(16) as p: res = np.array(p.map(sim_transistor, args, chunksize=100))
    out = pd.DataFrame(res, columns=["id0", "ig0", "is0", "ib0"])
    out.to_csv(path.replace(".csv", "_twin.csv"), index=False, float_format="%.10g")
    print(path, len(out), int(out.isna().any(axis=1).sum()), flush=True)

if __name__ == "__main__":
    for f in sys.argv[1:]: twin_file(f)
