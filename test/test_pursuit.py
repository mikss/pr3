import numpy as np
import pandas as pd

from pr3.pursuit import PracticalProjectionPursuitRegressor


def test_ppr(n_samples, p_dim):
    xcols = [f"f{i}" for i in range(p_dim)]
    ycol = "y"
    np.random.seed(2021)
    df = pd.DataFrame(np.random.normal(0, 1, (n_samples, p_dim)), columns=xcols)
    df["e"] = np.random.normal(0, 5, n_samples)
    df["y"] = (
        df["f0"] * df["f1"]
        + 1 / 2 * df["f2"] ** 2.0
        - 1 / 3 * df["f3"] ** 3.0
        + np.cos(2 * np.pi * df["f4"])
        + 0.5 * df["f5"]
        - 0.6 * df["f6"]
        + df["e"]
    )
    ppr = PracticalProjectionPursuitRegressor(
        n_stages=2, learning_rate=0.5, random_state=2021, stage_maxiter=5
    )
    ppr.fit(df[xcols], df[ycol])
