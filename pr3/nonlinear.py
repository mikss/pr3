from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Set, Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from statsmodels.nonparametric.kernel_regression import KernelReg


class UnivariateNonlinearRegressor(ABC):
    def fit(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> None:
        """Fits a univariate regressor."""
        for v in (x, y, w):
            self._validate_univariate(v)
        self._fit_univariate(x, y, w)

    @staticmethod
    def _validate_univariate(v: np.ndarray) -> None:
        """Ensures that model data is univariate."""
        one_dim = len(v.shape) == 1
        two_dim_one_col = (len(v.shape) == 2) and (v.shape[1] == 1)
        if not (one_dim or two_dim_one_col):
            raise ValueError("Univariate regression requires 1-D predictor and response.")

    @abstractmethod
    def _fit_univariate(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> None:
        """Trains a univariate model from data."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Outputs data inferences."""

    @staticmethod
    def weighted_resampler(
        x: np.ndarray, y: np.ndarray, w: np.ndarray, bootstrap_ratio: float = 4.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = w.shape[0]
        indices = np.random.choice(
            a=n_samples, size=int(bootstrap_ratio * n_samples), replace=True, p=w / w.sum(),
        )
        x = x[indices, :]
        y = y[indices, :]
        return x, y


class DecisionTreeUNLR(UnivariateNonlinearRegressor):
    decision_tree: DecisionTreeRegressor

    def __init__(self, max_depth: int = 8, min_weight_fraction_leaf: float = 1e-3):
        self.decision_tree = DecisionTreeRegressor(
            max_depth=max_depth, min_weight_fraction_leaf=min_weight_fraction_leaf, random_state=0
        )

    def _fit_univariate(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> None:
        self.decision_tree.fit(x, y, w)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.decision_tree.predict(x)


class PiecewiseLinearUNLR(UnivariateNonlinearRegressor):
    perceptron: MLPRegressor
    scale: float

    def __init__(
        self,
        components: int = 25,
        random_state: Optional[int] = None,
        max_iter: int = 2000,
        tol: float = 1e-4,
    ):
        self.perceptron = MLPRegressor(
            hidden_layer_sizes=(components,),
            activation="relu",
            solver="lbfgs",
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
        )

    def _fit_univariate(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> None:
        if w is not None:
            x, y = self.weighted_resampler(x, y, w)
        self.scale = y.std()
        self.perceptron.fit(x, y.ravel() / self.scale)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.perceptron.predict(x) * self.scale


class PolynomialUNLR(UnivariateNonlinearRegressor):
    degree: int
    polynomial: np.ndarray

    def __init__(self, degree: int = 6):
        self.degree = degree

    def _fit_univariate(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> None:
        self.polynomial = np.polyfit(
            x=x.ravel(), y=y.ravel(), deg=self.degree, w=np.sqrt(w.ravel()),
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.polyval(self.polynomial, x)


class NadarayaWatsonUNLR(UnivariateNonlinearRegressor):
    kernel: KernelReg
    bandwidth: float

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth

    def _fit_univariate(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> None:
        if w is not None:
            x, y = self.weighted_resampler(x, y, w)
        self.kernel = KernelReg(endog=y, exog=x, var_type="c", bw=[self.bandwidth])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.kernel.fit(x)[0]


class RidgeFunctionRegistry(Enum):
    tree = DecisionTreeUNLR
    piecewise = PiecewiseLinearUNLR
    polynomial = PolynomialUNLR
    kernel = NadarayaWatsonUNLR

    @classmethod
    def valid_mnemonics(cls) -> Set[str]:
        return set(name for name, _ in cls.__members__.items())

    @classmethod
    def valid_regressors(cls) -> Set[UnivariateNonlinearRegressor]:
        return set(value for _, value in cls.__members__.items())
