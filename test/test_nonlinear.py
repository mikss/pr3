import numpy as np
import pytest
from sklearn.metrics import r2_score

from pr3.nonlinear import (
    DecisionTreeUNLR,
    NadarayaWatsonUNLR,
    PiecewiseLinearUNLR,
    PolynomialUNLR,
    RidgeFunctionRegistry,
    UnivariateNonlinearRegressor,
)


def foo(x, poly_degree=4, trig_order=3):
    coefficients = np.random.normal(0, 1, poly_degree)
    exponents = list(range(poly_degree))
    poly = sum(coefficients[i] * x ** exponents[i] for i in range(poly_degree))

    amplitudes = np.random.exponential(25, trig_order)
    frequencies = np.random.exponential(0.25, trig_order)
    trig = sum(amplitudes[i] * np.cos(2 * np.pi * frequencies[i] * x) for i in range(trig_order))

    return poly + trig


@pytest.fixture()
def test_xyw(random_seed, n_samples, eps_std):
    np.random.seed(random_seed)
    eps = np.random.normal(0, eps_std, (n_samples, 1))
    x = np.random.normal(0, 1, (n_samples, 1))
    y = foo(x) + eps
    w = np.ones(n_samples)
    return x, y, w


def test_bad_xy(random_seed, n_samples):
    np.random.seed(random_seed)
    three_dim_x = np.random.normal(0, 1, (n_samples, 10, 10))
    multi_col_x = np.random.normal(0, 1, (n_samples, 2))
    for x in (three_dim_x, multi_col_x):
        with pytest.raises(ValueError):
            UnivariateNonlinearRegressor._validate_univariate(x)


@pytest.mark.parametrize(
    "regressor,init_kwargs,r2_threshold",
    [
        (DecisionTreeUNLR, dict(max_depth=4), 0.10),
        (PiecewiseLinearUNLR, dict(components=10), 0.05),
        (PolynomialUNLR, dict(degree=3), 0.04),
        (NadarayaWatsonUNLR, dict(bandwidth=1.0), 0.04),
    ],
)
def test_regression(test_xyw, regressor, init_kwargs, r2_threshold):
    x, y, w = test_xyw
    unlr = regressor(**init_kwargs)
    unlr.fit(x, y, w)
    y_hat = unlr.predict(x)
    assert r2_score(y, y_hat) > r2_threshold


def test_registry(registry_size=3):
    assert len(RidgeFunctionRegistry.valid_mnemonics()) == registry_size
    assert len(RidgeFunctionRegistry.valid_regressors()) == registry_size
