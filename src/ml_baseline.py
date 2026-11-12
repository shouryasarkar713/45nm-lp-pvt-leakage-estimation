"""The four required model families, each predicting every terminal current as sign x 10^(log10|I|)."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from ml_data import *

def make_reg(kind, seed=SEEDS["ml"]):
    if kind == "Ridge": return make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1e-2))
    if kind == "RandomForest": return RandomForestRegressor(n_estimators=120, min_samples_leaf=2, max_features=0.7, n_jobs=4, random_state=seed)
    if kind == "HistGradBoost": return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1e-3, random_state=seed)
    if kind == "MLP": return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", learning_rate_init=2e-3, batch_size=512, max_iter=40, early_stopping=True, n_iter_no_change=6, random_state=seed))
    raise ValueError(kind)

class TerminalModel:
    """Log-magnitude regressor(s) for the four terminals (+ HGB sign classifier where both signs occur).
    Ridge / RandomForest / MLP are native multi-output; HistGradientBoosting uses one regressor per terminal."""
    def __init__(self, kind): self.kind = kind; self.reg = None; self.sgn = {}
    def fit(self, X, Y):
        L = np.stack([lg(Y[t].values) for t in TERM], 1)
        if self.kind == "HistGradBoost": self.reg = [make_reg(self.kind).fit(X, L[:, k]) for k in range(4)]
        else: self.reg = make_reg(self.kind).fit(X, L)
        for t in TERM:
            s = np.sign(Y[t].values); s[s == 0] = 1
            if min((s > 0).mean(), (s < 0).mean()) > 1e-4:
                self.sgn[t] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEEDS["ml"]).fit(X, s > 0)
            else: self.sgn[t] = 1.0 if (s > 0).mean() > .5 else -1.0
        return self
    def predict(self, X):
        L = np.stack([r.predict(X) for r in self.reg], 1) if isinstance(self.reg, list) else self.reg.predict(X)
        out = {}
        for k, t in enumerate(TERM):
            sg = self.sgn[t]
            out[t] = 10 ** L[:, k] * (np.where(sg.predict(X), 1.0, -1.0) if hasattr(sg, "predict") else sg)
        return pd.DataFrame(out, index=X.index)
