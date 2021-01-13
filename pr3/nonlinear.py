from abc import ABC, abstractmethod
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class UnivariateNonlinearRegressor(ABC):
    trg_range: Tuple[float, float]

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fits a univariate regressor."""
        self.validate_univariate(x)
        self.validate_univariate(y)
        self.fit_univariate(x, y)

    @staticmethod
    def validate_univariate(v: np.ndarray) -> None:
        """Ensures that model data is univariate."""
        one_dim = len(v.shape) == 1
        two_dim_one_col = (len(v.shape) == 2) and (v.shape[1] == 1)
        if not (one_dim or two_dim_one_col):
            raise ValueError("Univariate regression requires 1-D predictor and response.")

    @abstractmethod
    def fit_univariate(self, x: np.ndarray, y: np.ndarray) -> None:
        """Trains a univariate model from data."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Outputs data inferences."""

    def plot(
        self,
        trg_x: Optional[np.ndarray],
        trg_y: Optional[np.ndarray],
        scatter_sample_ratio: float = 1.0,
        **subplot_kwargs,
    ) -> None:
        """Plots the learned one-dimensional function and (optionally) includes a scatter plot of training data."""
        fig, ax = plt.subplots(**subplot_kwargs)

        trg_span = self.trg_range[1] - self.trg_range[0]
        trg_grid = np.linspace(
            self.trg_range[0] - trg_span * 0.05, self.trg_range[1] + trg_span * 0.05
        )
        ridge_function = pd.Series(index=trg_grid, data=self.predict(trg_grid))
        ridge_function.plot(ax=ax, kind="line", linewidth=5, color="r")
        if all(data is not None for data in (trg_x, trg_y)):
            trg_data = pd.Series(index=trg_x, data=trg_y).sample(frac=scatter_sample_ratio)
            trg_data.plot(ax=ax, kind="scatter", marker=".", markersize=1, color="k", alpha=0.5)


# TODO complete the implementation of the "inner loop" nonlinear univariate iteration
# polynomial regression via numpy polyfit https://zerowithdot.com/polynomial-regression-in-python/
# spline regression via scipy UnivariateSpline https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
# decision tree via sklearn https://en.wikipedia.org/wiki/Decision_tree_learning
# kernel regression via statsmodels https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.kernel_regression.KernelReg.html
# rectifier via numba (should be able to be fully `nopython=True` and `nogil=True`)
# introduce test for the plot
