from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Set, Tuple, Union

import numpy as np
from scipy.stats import gennorm
from sklearn.linear_model import lars_path_gram
from sklearn.utils.validation import check_random_state


class ProjectionVector:
    beta: np.ndarray
    q: int

    def __init__(self, q: int):
        self.q = q

    def _normalize(self) -> None:
        try:
            self.beta = self.beta / np.linalg.norm(self.beta, ord=self.q, axis=0)
        except AttributeError:
            raise AttributeError("Must set attribute `beta` before normalizing.")


class ProjectionSampler(ProjectionVector):
    def __init__(
        self,
        p: int,
        q: int = 2,
        sparsity: int = -1,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        """Generates a normalized random projection vector (for initialization purposes).

        Args:
            p: The dimension of the vector.
            q: The order of ell^q unit ball from which to sample.
            sparsity: The number of non-zero coordinates; pass -1 for a dense vector.
            random_state: NumPy random state.
        """
        super().__init__(q=q)
        _rs = check_random_state(random_state)
        if sparsity > 0:
            q_generalized_normal = np.zeros((p, 1))
            idxs = _rs.choice(a=p, size=sparsity, replace=False)
            q_generalized_normal[idxs, 0] = gennorm.rvs(beta=q, size=sparsity)
        else:
            q_generalized_normal = gennorm.rvs(beta=q, size=(p, 1))
        self.beta = q_generalized_normal
        self._normalize()


class SufficientStatisticsRegressionProjection(ABC, ProjectionVector):
    def fit(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> None:
        """Fits weighted least squares (WLS) linear regression model via sufficient statistics.

        Args:
            x: design matrix of shape (n_samples, n_features)
            y: response matrix of shape (n_samples, n_responses)
            w: weight vector of shape (n_samples,)
        """
        w = self._reshape_weights(x.shape[0], w)
        xtx, xty = self.compute_sufficient_statistics(x, y, w)
        wess = self.compute_effective_sample_size(w)
        self._fit_sufficient(xtx, xty, wess)

    def fit_normalize(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> None:
        self.fit(x, y, w)
        self._normalize()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts from linear model."""
        return x @ self.beta

    @staticmethod
    def _reshape_weights(n: int, w: Optional[np.ndarray] = None) -> np.ndarray:
        if w is None:
            w = np.ones(n)
        return w.reshape((n, 1))

    @staticmethod
    def compute_sufficient_statistics(
        x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes (weighted) sufficient statistics for WLS regression (Gram xtx, cross xty)."""
        xtx = np.multiply(x, w).T @ x / w.sum()
        xty = np.multiply(x, w).T @ y / w.sum()
        return xtx, xty

    @staticmethod
    def compute_effective_sample_size(w: np.ndarray) -> float:
        """Computes effective sample size given sample weights."""
        return w.sum() ** 2.0 / (w ** 2.0).sum()

    @abstractmethod
    def _fit_sufficient(self, xtx, xty, wess) -> None:
        """Fits linear model using only second-order sufficient statistics."""


class LowerUpperRegressionProjection(SufficientStatisticsRegressionProjection):
    ridge: float

    def __init__(self, q: int = 2, ridge: float = 0.0):
        """Instantiates a WLS linear model with ridge regularization and q-normalized beta.

        This implementation computes regression coefficients by solving a system of linear equations
        via the LU decomposition, which is the technique implemented by `gesv`, the LAPACK routine
        called by `np.linalg.solve`.

        Args:
            q: The order of ell^q norm with which to normalize resultant beta.
            ridge: Regularization level.
        """
        super().__init__(q=q)
        self.ridge = ridge

    def _fit_sufficient(self, xtx, xty, wess) -> None:
        self.beta = np.linalg.solve(xtx + self.ridge * np.eye(xtx.shape[0]), xty)


class LeastAngleRegressionProjection(SufficientStatisticsRegressionProjection):
    max_iter: int
    min_corr: float
    alpha: np.ndarray

    def __init__(self, q: int = 2, max_iter: int = 100, min_corr: float = 1e-4):
        """Instantiates a WLS linear regression model with sparse and q-normalized beta.

        This implementation computes regression coefficients by iteratively traversing the LASSO
        regularization path, which serves as an efficient way to solve the l1-regularized least
        squares optimization problem (on par with, say, FISTA, but typically more efficient than
        black-box quadratic programming methods).

        Args:
            q: The order of ell^q norm with which to normalize resultant beta.
            max_iter: Maximum number of iterations.
        """
        super().__init__(q=q)
        self.max_iter = max_iter
        self.min_corr = min_corr

    def _fit_sufficient(self, xtx, xty, wess) -> None:
        self.alpha, _, coefs = lars_path_gram(
            Xy=xty[:, 0],
            Gram=xtx,
            n_samples=wess,
            max_iter=self.max_iter,
            alpha_min=self.min_corr,
            method="lasso",
        )
        self.beta = coefs[:, [-1]]


class ProjectionOptimizerRegistry(Enum):
    lower_upper = LowerUpperRegressionProjection
    least_angle = LeastAngleRegressionProjection

    @classmethod
    def valid_mnemonics(cls) -> Set[str]:
        return set(name for name, _ in cls.__members__.items())

    @classmethod
    def valid_regressors(cls) -> Set[SufficientStatisticsRegressionProjection]:
        return set(value for _, value in cls.__members__.items())
