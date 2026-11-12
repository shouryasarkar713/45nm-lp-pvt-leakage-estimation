"""Fixed 70/15/15 split of each 10,000-row base dataset (seed 2024)."""
import numpy as np
from config import *
def split_idx(n=N_SPEC, seed=SEEDS["split"]):
    perm = np.random.default_rng(seed).permutation(n)
    return perm[: int(.7 * n)], perm[int(.7 * n): int(.85 * n)], perm[int(.85 * n):]
