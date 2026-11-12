"""Writes, per case, the hashes of the key columns of every training row (and the val/test rows), so that the absence of
validation/test rows from the training table can be re-verified without the (large) supplementary CSV files."""
import os, numpy as np, pandas as pd
from config import *
from ml_data import load_train, load_val_test
KEY = ["vdd", "temp", "w", "vd", "vs", "vth0", "toxe", "ndep", "u0", "xj"]
def hashes(df): return pd.util.hash_pandas_object(df[KEY].round(12), index=False).values.astype(np.uint64)
if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    for c in CASES:
        tr = load_train(c, extra=True); va, te = load_val_test(c)
        np.savez_compressed(os.path.join(OUT, "tables", f"row_hashes_{c}.npz"), train=hashes(tr), val=hashes(va), test=hashes(te), train_src=tr.src.values.astype("U12"))
        print(c, len(tr), len(set(hashes(tr)) & (set(hashes(va)) | set(hashes(te)))))
