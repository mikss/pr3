from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from sklearn.tree import DecisionTreeRegressor


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

    # # TODO move this upstream?
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
    #     ridge_function = pd.Series(index=trg_grid, data=self.predict(trg_grid.reshape((-1, 1))))
    #     ridge_function.plot(ax=ax, kind="line", linewidth=5, color="r")
    #     if all(data is not None for data in (trg_x, trg_y)):
    #         self._validate_univariate(trg_x)
    #         self._validate_univariate(trg_y)
    #         trg_data = pd.Series(index=trg_x.reshape(-1), data=trg_y.reshape(-1)).sample(
    #             frac=scatter_sample_ratio
    #         )
    #         trg_data.plot(ax=ax, marker=".", markersize=1, linestyle="", color="k", alpha=0.5)


class DecisionTreeUNLR(UnivariateNonlinearRegressor):
    decision_tree: DecisionTreeRegressor

    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 10, random_state: int = 0):
        self.decision_tree = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state
        )

    def _fit_univariate(self, x: np.ndarray, y: np.ndarray) -> None:
        self.decision_tree.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.decision_tree.predict(x)


# TODO complete the implementation of the "inner loop" nonlinear univariate iteration
# polynomial regression via numpy polyfit https://zerowithdot.com/polynomial-regression-in-python/
# spline regression via scipy UnivariateSpline https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
# kernel regression via statsmodels https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.kernel_regression.KernelReg.html
# rectifier via numba (should be able to be fully `nopython=True` and `nogil=True`)
# introduce test for the plot
