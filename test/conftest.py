import pytest


@pytest.fixture
def random_seed():
    return 2021


@pytest.fixture
def p_dim():
    return 100


@pytest.fixture
def q_dim():
    return 2


@pytest.fixture
def sparsity():
    return 10


@pytest.fixture
def n_samples():
    return 10000


@pytest.fixture
def eps_std():
    return 100
