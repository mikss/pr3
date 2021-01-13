import numpy as np
import pytest
from sklearn.metrics import r2_score

from pr3.linear import (
    LeastAngleRegressionProjection,
    LowerUpperRegressionProjection,
    ProjectionSampler,
    ProjectionVector,
)


def test_normalize(random_seed, p_dim, q_dim):
    np.random.seed(random_seed)
    projection = ProjectionVector(q=q_dim)
    with pytest.raises(AttributeError):
        projection._normalize()
    projection.beta = np.random.normal(0, 1, (p_dim,))
    projection._normalize()
    np.testing.assert_almost_equal(np.linalg.norm(projection.beta, ord=q_dim), 1, decimal=15)


def test_sampler(random_seed, p_dim, q_dim, sparsity=5):
    np.random.seed(random_seed)
    sparse_projection = ProjectionSampler(p=p_dim, q=q_dim, sparsity=sparsity)
    dense_projection = ProjectionSampler(p=p_dim, q=q_dim)

    np.testing.assert_almost_equal(np.linalg.norm(sparse_projection.beta, ord=q_dim), 1, decimal=15)
    np.testing.assert_almost_equal(np.linalg.norm(dense_projection.beta, ord=q_dim), 1, decimal=15)
    assert np.count_nonzero(sparse_projection.beta) == sparsity
    assert np.count_nonzero(dense_projection.beta) == p_dim


@pytest.fixture()
def test_xy(random_seed, p_dim, q_dim, sparsity, n_samples, eps_std):
    np.random.seed(random_seed)
    eps = np.random.normal(0, eps_std, (n_samples, 1))
    beta = np.zeros((p_dim, 1))
    beta[:sparsity, :] = ProjectionSampler(p=sparsity, q=q_dim, seed=random_seed).beta
    x = np.random.normal(0, 1, (n_samples, p_dim))
    y = x @ beta + eps
    return x, y


@pytest.mark.parametrize(
    "regressor,init_kwargs,r2_threshold",
    [
        (LowerUpperRegressionProjection, dict(ridge=1.0), 19e-4),
        (LeastAngleRegressionProjection, dict(max_iter=25, min_corr=1e-4), 14e-4),
    ],
)
def test_regression(test_xy, regressor, init_kwargs, r2_threshold):
    x, y = test_xy
    lurp = regressor(**init_kwargs)
    lurp.fit_normalize(x, y)
    y_hat = lurp.predict(x)
    assert r2_score(y, y_hat) > r2_threshold
