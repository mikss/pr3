from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Set, Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from statsmodels.nonparametric.kernel_regression import KernelReg


class UnivariateNonlinearRegressor(ABC):
    _trg_range: Tuple[float, float]

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fits a univariate regressor."""
        self._validate_univariate(x)
        self._validate_univariate(y)
        self._trg_range = (x.min(), x.max())
        self._fit_univariate(x, y)

    @staticmethod
    def _validate_univariate(v: np.ndarray) -> None:
        """Ensures that model data is univariate."""
        one_dim = len(v.shape) == 1
        two_dim_one_col = (len(v.shape) == 2) and (v.shape[1] == 1)
        if not (one_dim or two_dim_one_col):
            raise ValueError("Univariate regression requires 1-D predictor and response.")

    @abstractmethod
    def _fit_univariate(self, x: np.ndarray, y: np.ndarray) -> None:
        """Trains a univariate model from data."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Outputs data inferences."""

    # # TODO move this upstream? introduce a test while mocking plot capability
    # def plot(
    #     self,
    #     trg_x: Optional[np.ndarray] = None,
    #     trg_y: Optional[np.ndarray] = None,
    #     scatter_sample_ratio: float = 0.1,
    #     **subplot_kwargs,
    # ) -> None:
    #     """Plots the learned one-dimensional function and (optionally) includes a scatter plot of training data."""
    #     fig, ax = plt.subplots(**subplot_kwargs)
    #
    #     trg_span = self._trg_range[1] - self._trg_range[0]
    #     trg_grid = np.linspace(
    #         start=self._trg_range[0] - trg_span * 0.05,
    #         stop=self._trg_range[1] + trg_span * 0.05,
    #         num=10000,
    #     )
    #     ridge_function = pd.Series(index=trg_grid, data=self.predict(trg_grid.ravel()))
    #     ridge_function.plot(ax=ax, kind="line", linewidth=5, color="r")
    #     if all(data is not None for data in (trg_x, trg_y)):
    #         self._validate_univariate(trg_x)
    #         self._validate_univariate(trg_y)
    #         trg_data = pd.Series(index=trg_x.ravel(), data=trg_y.ravel()).sample(
    #             frac=scatter_sample_ratio
    #         )
    #         trg_data.plot(ax=ax, marker=".", markersize=1, linestyle="", color="k", alpha=0.5)


class DecisionTreeUNLR(UnivariateNonlinearRegressor):
    decision_tree: DecisionTreeRegressor

    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 10):
        self.decision_tree = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0
        )

    def _fit_univariate(self, x: np.ndarray, y: np.ndarray) -> None:
        self.decision_tree.fit(x, y)

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

    def _fit_univariate(self, x: np.ndarray, y: np.ndarray) -> None:
        self.scale = y.std()
        self.perceptron.fit(x, y.ravel() / self.scale)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.perceptron.predict(x) * self.scale


class PolynomialUNLR(UnivariateNonlinearRegressor):
    degree: int
    polynomial: np.ndarray

    def __init__(
        self, degree: int = 6,
    ):
        self.degree = degree

    def _fit_univariate(self, x: np.ndarray, y: np.ndarray) -> None:
        self.polynomial = np.polyfit(x.ravel(), y.ravel(), self.degree)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.polyval(self.polynomial, x)


class NadarayaWatsonUNLR(UnivariateNonlinearRegressor):
    kernel: KernelReg
    bandwidth: float

    def __init__(
        self, bandwidth: float = 1,
    ):
        self.bandwidth = bandwidth

    def _fit_univariate(self, x: np.ndarray, y: np.ndarray) -> None:
        self.kernel = KernelReg(endog=y, exog=x, var_type="c", bw=[self.bandwidth])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.kernel.fit(x)[0]


class RidgeFunctionRegistry(Enum):
    tree: DecisionTreeUNLR
    piecewise: PiecewiseLinearUNLR
    polynomial: PolynomialUNLR
    kernel: NadarayaWatsonUNLR

    @classmethod
    def valid_mnemonics(cls) -> Set[str]:
        return set(name for name, _ in cls.__members__.items())

    @classmethod
    def valid_regressors(cls) -> Set[UnivariateNonlinearRegressor]:
        return set(value for _, value in cls.__members__.items())
