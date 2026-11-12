import numpy as np, pandas as pd
from gen_transistor import draw, simulate
from config import *
if __name__ == "__main__":
    for case, seed in zip(CASES, SEEDS["spec"]):
        df = draw(case, N_SPEC, np.random.default_rng(seed)).iloc[:300]; new = simulate(case, df)
        old = pd.read_csv(os.path.join(DATA, "base", f"{case}.csv")).iloc[:300]
        inp = np.allclose(new.iloc[:, :12].values, old.iloc[:, :12].values, rtol=1e-8)
        cur = np.nanmax(np.abs(new[["id","ig","is","ib"]].values / old[["id","ig","is","ib"]].values - 1)[np.abs(old[["id","ig","is","ib"]].values) > 1e-15])
        print(case, "inputs identical:", inp, " max relative current difference (|I|>1e-15):", cur)
