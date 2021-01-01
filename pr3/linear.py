from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from scipy.stats import gennorm


class ProjectionVector:
    beta: np.ndarray

    @staticmethod
    def _normalize(v: np.ndarray, q: int = 2) -> np.ndarray:
        return v / np.linalg.norm(v, ord=q)


class ProjectionSampler(ProjectionVector):
    def __init__(self, p: int, q: int = 2, sparsity: int = -1, seed: Optional[int] = None):
        """Generate a normalized random projection vector (for initialization purposes).

        Args:
            p: The dimension of the vector.
            q: The order of ell^q unit ball from which to sample.
            sparsity: The number of non-zero coordinates; pass -1 for a dense vector.
            seed: NumPy random seed.
        """
        np.random.seed(seed)
        if sparsity > 0:
            q_generalized_normal = np.zeros(p)
            idxs = np.random.choice(a=p, size=sparsity, replace=False)
            q_generalized_normal[idxs] = gennorm.rvs(beta=q, size=sparsity)
        else:
            q_generalized_normal = gennorm.rvs(beta=q, size=p)
        self.beta = self._normalize(v=q_generalized_normal, q=q)


class WeightedLeastSquaresSufficientStatistics(ABC):
    # TODO separate tests for each method
    beta: np.ndarray

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        """Fit weighted least squares linear regression model by first computing sufficient statistics.

        Args:
            x: design matrix of shape (n_samples, n_features)
            y: response matrix of shape (n_samples, n_responses)
            w: weight vector of shape (n_samples,)
        """
        xtx, xty = self.compute_sufficient_statistics(x, y, w)
        self._fit_sufficient(xtx, xty)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict from linear model."""
        return x @ self.beta

    @staticmethod
    def compute_sufficient_statistics(
        x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sufficient statistics for least squares regression (Gram matrix xtx, cross matrix xty)."""
        xtx = np.multiply(x, w).T @ x
        xty = np.multiply(x, w).T @ y
        return xtx, xty

    @abstractmethod
    def _fit_sufficient(self, xtx, xty):
        # TODO implement both ordinary linear algebraic method (least squares) and sparse method (LARS, iterative)
        #  both with normalization, both inheriting this and the other
        ...
