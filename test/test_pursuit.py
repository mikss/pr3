import numpy as np
import pandas as pd
import pytest

from pr3.pursuit import PracticalProjectionPursuitRegressor


def test_ppr_fit(n_samples, p_dim):
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

    # high noise
    ppr = PracticalProjectionPursuitRegressor(
        n_stages=2, learning_rate=0.5, random_state=2021, stage_maxiter=5
    )
    ppr.fit(df[xcols], df[ycol])
    _ = ppr.predict(df[xcols])
    ppr.plot_losses()
    with pytest.raises(ValueError):
        ppr.plot(df[xcols].values, df[ycol].values, feature_names=["a"])
    ppr.plot(df[xcols].values, df[ycol].values)

    # low noise (early stop)
    df["y"] = df["f0"]
    ppr = PracticalProjectionPursuitRegressor(
        n_stages=2, learning_rate=1.0, random_state=2021, stage_maxiter=50, stage_tol=1e-2,
    )
    ppr.fit(df[["f0"]], df[ycol])


def test_ppr_init():
    with pytest.raises(ValueError):
        PracticalProjectionPursuitRegressor(n_stages=-1)
    with pytest.raises(ValueError):
        PracticalProjectionPursuitRegressor(learning_rate=2.0)
    with pytest.raises(ValueError):
        PracticalProjectionPursuitRegressor(ridge_function_class="blahblah")
    with pytest.raises(ValueError):
        PracticalProjectionPursuitRegressor(projection_optimizer_class="blahblah")
    with pytest.raises(ValueError):
        PracticalProjectionPursuitRegressor(stage_tol=-1e-4)
    with pytest.raises(ValueError):
        PracticalProjectionPursuitRegressor(stage_maxiter=0)
