"""numpy-only inference of an exported StructModel (no torch needed to use the saved models)."""
import numpy as np, pandas as pd
from ml_data import *
HEADS = ["id0", "g", "ig", "ib"]

class StructInference:
    """numpy-only inference of an exported StructModel (no torch needed to use the saved model)."""
    def __init__(self, e): self.__dict__.update(e)
    def predict(self, d):
        X = (features(d, self.case).values.astype(np.float32) - self.mu) / self.sd
        rs = []; ss = []
        for n in self.nets:
            h = X
            for W, b in n["W"]: z = h @ W.T + b; h = z / (1 + np.exp(-np.clip(z, -60, 60)))
            rs.append(h @ n["Wr"][0].T + n["Wr"][1]); ss.append(h @ n["Ws"][0].T + n["Ws"][1])
        r = np.mean(rs, 0); s = np.mean(ss, 0)
        out = {}
        for k, hd in enumerate(HEADS):
            mag = 10 ** (r[:, k] * self.tsd[hd] + self.tmu[hd])
            pos = (s[:, k] >= 0) if self.sconst[hd] is None else np.full(len(d), self.sconst[hd] > 0.5)
            out[hd] = (mag, np.where(pos, 1.0, -1.0))
        vds = (d.vd - d.vs).values
        R = np.sign(vds) * out["g"][1] * out["g"][0] * np.abs(vds)
        idp = out["id0"][0] * out["id0"][1] + R; igp = out["ig"][0] * out["ig"][1]; ibp = out["ib"][0] * out["ib"][1]
        return pd.DataFrame({"id": idp, "ig": igp, "is": -idp - igp - ibp, "ib": ibp}, index=d.index)
