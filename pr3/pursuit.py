from __future__ import annotations

from typing import Dict, List, Optional, Type, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, check_X_y

from pr3.linear import (
    ProjectionOptimizerRegistry,
    ProjectionSampler,
    ProjectionVector,
    SufficientStatisticsRegressionProjection,
)
from pr3.nonlinear import RidgeFunctionRegistry, UnivariateNonlinearRegressor


# TODO: many docstrings need revision or creation here
# TODO: introduce tests
class PracticalProjectionPursuitRegressor(BaseEstimator, TransformerMixin, RegressorMixin):

    projections: List[Optional[SufficientStatisticsRegressionProjection]]
    ridge_functions: List[Optional[UnivariateNonlinearRegressor]]
    loss_path: List[List[float]]
    _rs: np.random.RandomState

    def __init__(
        self,
        n_stages: int = 50,
        learning_rate: float = 0.25,
        ridge_function_class: Union[str, Type[UnivariateNonlinearRegressor]] = "piecewise",
        ridge_function_kwargs: Optional[Dict] = None,
        projection_init_class: Type[ProjectionVector] = ProjectionSampler,
        projection_init_kwargs: Optional[Dict] = None,
        projection_optimizer_class: Type[SufficientStatisticsRegressionProjection] = "least_angle",
        projection_optimizer_kwargs: Optional[Dict] = None,
        stage_tol: float = 1e-8,
        stage_maxiter: int = 128,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        if n_stages < 1:  # TODO augment to interpret negative to be "as many as necessary".
            raise ValueError("Require `n_stages` >= 1.")
        if not learning_rate <= 1.0 and learning_rate > 0:
            raise ValueError("Learning rate must lie in (0, 1].")
        if not (
            ridge_function_class
            in RidgeFunctionRegistry.valid_mnemonics() | RidgeFunctionRegistry.valid_regressors()
        ):
            raise ValueError(f"Invalid ridge function class {ridge_function_class}.")
        if not (
            projection_optimizer_class
            in ProjectionOptimizerRegistry.valid_mnemonics()
            | ProjectionOptimizerRegistry.valid_regressors()
        ):
            raise ValueError(f"Invalid projection optimizer class {projection_optimizer_class}.")
        if stage_tol < 0:
            raise ValueError("Require `stage_tol` >= 0.")
        if stage_maxiter < 1:
            raise ValueError("Require `stage_maxiter` >= 1.")

        self.n_stages = n_stages
        self.ridge_function_class = ridge_function_class
        self.ridge_function_kwargs = ridge_function_kwargs
        self.projection_init_class = projection_init_class  # TODO allow "warm start"?
        self.projection_init_kwargs = projection_init_kwargs
        self.projection_optimizer_class = projection_optimizer_class
        self.projection_optimizer_kwargs = projection_optimizer_kwargs
        self.stage_tol = stage_tol
        self.stage_maxiter = stage_maxiter

        self._rs = check_random_state(random_state)

    def fit(
        self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
    ) -> PracticalProjectionPursuitRegressor:
        x, y, w = self._sklearn_validate(x, y, w)
        self._initialize_stages(feature_dim=x.shape[1])
        for stage in range(self.n_stages):
            self._fit_stage(x, y, w, stage)
        return self

    @staticmethod
    def _sklearn_validate(x: np.ndarray, y: np.ndarray, w: np.ndarray):
        x, y = check_X_y(x, y, multi_output=False)
        y = y.reshape((-1, 1))
        w = _check_sample_weight(w, x, dtype=x.dtype)
        return x, y, w

    def _initialize_stages(self, feature_dim: int):
        self.projections = [None for _ in range(self.n_stages)]
        self.ridge_functions = [None for _ in range(self.n_stages)]
        self.loss_path = [[] for _ in range(self.n_stages)]

    def _fit_stage(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, stage: int):
        residuals = y - sum(
            [self.ridge_functions[j].predict(self.transform(x)[:, j]) for j in range(stage)]
        )

        # track multiple elements through alternating minimization
        beta = self.projection_init_class(**self.projection_init_kwargs)
        ridge_function = self.ridge_function_class(**self.ridge_function_kwargs)
        projection_optimizer = self.projection_optimizer_class(**self.projection_optimizer_kwargs)
        loss = np.inf
        for i in range(self.stage_maxiter):
            # optimize nonlinear ridge function
            ridge_function.fit(x @ beta, residuals)
            f = ridge_function.predict
            df = ridge_function.derivative

            # optimize linear projection
            projection_optimizer.fit_normalize(
                x=x,
                y=(x @ beta) + (residuals - f(x @ beta)) / df(x @ beta),
                w=w * df(x @ beta) ** 2.0,
            )
            beta = projection_optimizer.beta

            # update loss variables
            prev_loss = loss
            loss = np.sum(w * (residuals - f(x @ beta)) ** 2.0)
            self.loss_path[stage].append(loss)
            if abs(prev_loss - loss) < self.stage_tol:
                break

        self.projections[stage] = projection_optimizer
        self.ridge_functions[stage] = ridge_function

    def transform(self, x: np.ndarray):
        check_is_fitted(self, ["projections", "ridge_functions"])
        x = check_array(x)
        return np.array(x) @ np.concatenate(
            [self.projections[stage].beta for stage in range(self.n_stages)], axis=1,
        )

    def predict(self, x: np.ndarray):
        projected = self.transform(x)
        yhat = sum(
            (
                self.ridge_functions[stage].predict(projected[:, stage])
                for stage in range(self.n_stages)
            )
        )
        return yhat

    # TODO fix plotting functionality
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
    # _trg_range: Tuple[float, float]
    #     self._trg_range = (x.min(), x.max())
    # TODO introduce capability for plotting losses (with stages demarcated)
