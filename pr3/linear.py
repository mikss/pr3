from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from scipy.stats import gennorm

# from sklearn.linear_model import lars_path_gram


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
    def __init__(self, p: int, q: int = 2, sparsity: int = -1, seed: Optional[int] = None):
        """Generate a normalized random projection vector (for initialization purposes).

        Args:
            p: The dimension of the vector.
            q: The order of ell^q unit ball from which to sample.
            sparsity: The number of non-zero coordinates; pass -1 for a dense vector.
            seed: NumPy random seed.
        """
        super().__init__(q=q)
        np.random.seed(seed)
        if sparsity > 0:
            q_generalized_normal = np.zeros((p, 1))
            idxs = np.random.choice(a=p, size=sparsity, replace=False)
            q_generalized_normal[idxs, 0] = gennorm.rvs(beta=q, size=sparsity)
        else:
            q_generalized_normal = gennorm.rvs(beta=q, size=(p, 1))
        self.beta = q_generalized_normal
        self._normalize()


class SufficientStatisticsRegressionProjection(ABC, ProjectionVector):
    def fit(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> None:
        """Fit weighted least squares linear regression model by first computing sufficient statistics.

        Args:
            x: design matrix of shape (n_samples, n_features)
            y: response matrix of shape (n_samples, n_responses)
            w: weight vector of shape (n_samples,)
        """
        w = self._reshape_weights(x.shape[0], w)
        xtx, xty = self.compute_sufficient_statistics(x, y, w)
        self._fit_sufficient(xtx, xty)

    def fit_normalize(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> None:
        self.fit(x, y, w)
        self._normalize()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict from linear model."""
        return x @ self.beta

    @staticmethod
    def _reshape_weights(n: int, w: Optional[np.ndarray] = None):
        if w is None:
            w = np.ones(n)
        return w.reshape((n, 1))

    @staticmethod
    def compute_sufficient_statistics(
        x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sufficient statistics for least squares regression (Gram matrix xtx, cross matrix xty)."""
        xtx = np.multiply(x, w).T @ x / w.sum()
        xty = np.multiply(x, w).T @ y / w.sum()
        return xtx, xty

    @abstractmethod
    def _fit_sufficient(self, xtx, xty):
        """Fit linear model using only second-order sufficient statistics."""


class LowerUpperRegressionProjection(SufficientStatisticsRegressionProjection):
    ridge: float

    def __init__(self, q: int = 2, ridge: float = 0):
        """Instantiate a weighted least squares linear regression model with ridge regularization and q-normalized beta.

        This implementation computes regression coefficients by solving a system of linear equations via the LU
        decomposition, which is the technique implemented by `gesv`, the LAPACK routine called by `np.linalg.solve`.

        Args:
            q: The order of ell^q norm with which to normalize resultant beta.
            ridge: Regularization level.
        """
        super().__init__(q=q)
        self.ridge = ridge

    def _fit_sufficient(self, xtx, xty):
        self.beta = np.linalg.solve(xtx + self.ridge * np.eye(xtx.shape[0]), xty)


class LeastAngleRegressionProjection(SufficientStatisticsRegressionProjection):
    max_iter: int

    def __init__(self, q: int = 2, max_iter: int = 0):
        """Instantiate a weighted least squares linear regression model with sparse and q-normalized beta.

        This implementation computes regression coefficients by iteratively traversing the LASSO regularization path,
        which serves as an efficient way to solve the l1-regularized least squares optimization problem (on par with,
        say, FISTA, but typically more efficient than black-box quadratic programming methods).

        Args:
            q: The order of ell^q norm with which to normalize resultant beta.
            max_iter: Maximum number of iterations.
        """
        super().__init__(q=q)
        self.max_iter = max_iter

    def _fit_sufficient(self, xtx, xty):
        ...  # TODO using `lars_path_gram`, and add test
