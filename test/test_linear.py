import numpy as np

from pr3.linear import ProjectionSampler, ProjectionVector


def test_normalize(random_seed, p=100, q=2):
    np.random.seed(random_seed)
    v = np.random.normal(0, 1, (p,))
    projection = ProjectionVector()
    projection.beta = v
    projection._normalize(q=q)
    np.testing.assert_almost_equal(np.linalg.norm(projection.beta, ord=q), 1, decimal=15)


def test_sampler(random_seed, p=100, q=2, sparsity=5):
    np.random.seed(random_seed)
    sparse_projection = ProjectionSampler(p=p, q=q, sparsity=sparsity)
    dense_projection = ProjectionSampler(p=p, q=q)

    np.testing.assert_almost_equal(np.linalg.norm(sparse_projection.beta, ord=q), 1, decimal=15)
    np.testing.assert_almost_equal(np.linalg.norm(dense_projection.beta, ord=q), 1, decimal=15)
    assert np.count_nonzero(sparse_projection.beta) == sparsity
    assert np.count_nonzero(dense_projection.beta) == p
