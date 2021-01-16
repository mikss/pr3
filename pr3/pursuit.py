from __future__ import annotations

import inspect
import logging
from typing import Dict, List, Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, check_X_y

from pr3.linear import (
    ProjectionOptimizerRegistry,
    ProjectionSampler,
    SufficientStatisticsRegressionProjection,
)
from pr3.nonlinear import RidgeFunctionRegistry, UnivariateNonlinearRegressor


class PracticalProjectionPursuitRegressor(BaseEstimator, TransformerMixin, RegressorMixin):

    projections: List[Optional[SufficientStatisticsRegressionProjection]]
    ridge_functions: List[Optional[UnivariateNonlinearRegressor]]
    loss_path: List[List[float]]
    _rs: np.random.RandomState

    def __init__(
        self,
        n_stages: int = 10,
        learning_rate: float = 0.25,
        projection_init_sparsity: int = -1,
        ridge_function_class: Union[str, Type[UnivariateNonlinearRegressor]] = "piecewise",
        ridge_function_kwargs: Optional[Dict] = None,
        projection_optimizer_class: Union[
            str, Type[SufficientStatisticsRegressionProjection]
        ] = "least_angle",
        projection_optimizer_kwargs: Optional[Dict] = None,
        stage_tol: float = 1e-8,
        stage_maxiter: int = 128,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        """Implements projection pursuit regression.

        Args:
            n_stages: number of projection steps
            learning_rate: fraction by which to apply the result of each stage (a la boosting)
            ridge_function_class: nonlinear model to apply to each projected variable
            ridge_function_kwargs: keyword args to pass to `ridge_function_class`
            projection_init_sparsity: the number of non-zero elements for projection initialization
            projection_optimizer_class: linear model with which to train projection vectors
            projection_optimizer_kwargs: keyword args to pass to `projection_optimizer_kwargs`
            stage_tol: tolerance level at which to stop iterations within a stage
            stage_maxiter: maximum number of iterations within a stage
            random_state: random state to pass through to internal methods

        References:
            J.H. Friedman and W. Stuetzle,
             https://www.tandfonline.com/doi/abs/10.1080/01621459.1981.10477729
        """
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
        self.learning_rate = learning_rate
        self.ridge_function_class = (
            RidgeFunctionRegistry[ridge_function_class].value
            if isinstance(ridge_function_class, str)
            else ridge_function_class
        )
        self.ridge_function_kwargs = ridge_function_kwargs or dict()
        self.projection_sparsity = projection_init_sparsity  # TODO allow "warm start"?
        self.projection_optimizer_class = (
            ProjectionOptimizerRegistry[projection_optimizer_class].value
            if isinstance(projection_optimizer_class, str)
            else projection_optimizer_class
        )
        self.projection_optimizer_kwargs = projection_optimizer_kwargs or dict()
        self.stage_tol = stage_tol
        self.stage_maxiter = stage_maxiter

        self._rs = check_random_state(random_state)
        if ("random_state" not in self.ridge_function_kwargs.keys()) and (
            "random_state"
            in inspect.signature(self.ridge_function_class.__init__).parameters.keys()
        ):
            self.ridge_function_kwargs["random_state"] = self._rs

    def fit(
        self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
    ) -> PracticalProjectionPursuitRegressor:
        """Fits the PPR model stagewise."""
        x, y, w = self._sklearn_validate(x, y, w)
        self._initialize_stages()
        for stage in range(self.n_stages):
            logging.info(f"Entering stage {stage}.")
            self._fit_stage(x, y, w, stage)
        return self

    @staticmethod
    def _sklearn_validate(x: np.ndarray, y: np.ndarray, w: np.ndarray):
        """Ensures training data is well formed."""
        x, y = check_X_y(x, y, multi_output=False)
        y = y.reshape((-1, 1))
        w = _check_sample_weight(w, x, dtype=x.dtype)
        return x, y, w

    def _initialize_stages(self):
        """Initializes model components for tracking."""
        self.projections = [None for _ in range(self.n_stages)]
        self.ridge_functions = [None for _ in range(self.n_stages)]
        self.loss_path = [[] for _ in range(self.n_stages)]

    def _fit_stage(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, stage: int):
        """Fits one stage of PPR via alternating minimization."""

        # learning rate acts as a "step size" parameter by training only against "partial" residuals
        hat = sum(
            [
                self.ridge_functions[j].predict(self.transform(x)[:, [j]]).reshape(-1, 1)
                for j in range(stage)
            ]
        )
        residuals = y - self.learning_rate * hat

        # track multiple elements through alternating minimization
        beta = ProjectionSampler(
            p=x.shape[1], sparsity=self.projection_sparsity, random_state=self._rs
        ).beta
        ridge_function = self.ridge_function_class(**self.ridge_function_kwargs)
        projection_optimizer = self.projection_optimizer_class(**self.projection_optimizer_kwargs)
        loss = np.inf
        for i in range(self.stage_maxiter):
            # optimize nonlinear ridge function
            ridge_function.fit(x @ beta, residuals)
            f = ridge_function.predict
            df = ridge_function.derivative

            # optimize linear projection
            taylor_y = (
                (x @ beta).ravel() + (residuals.ravel() - f(x @ beta)) / df(x @ beta)
            ).reshape(-1, 1)
            taylor_w = w * (df(x @ beta) ** 2.0).ravel()
            projection_optimizer.fit_normalize(
                x=x, y=taylor_y, w=taylor_w,
            )
            beta = projection_optimizer.beta

            # update loss variables
            prev_loss = loss
            loss = np.sum(w * (residuals.ravel() - f(x @ beta)) ** 2.0)
            self.loss_path[stage].append(loss)
            logging.info(
                f"Stage {stage}, iteration {i + 1} of max {self.stage_maxiter}, loss of {loss}."
            )
            if abs(prev_loss - loss) < self.stage_tol:
                logging.info(f"Early stopping stage {stage} at iteration {i}.")
                break

        self.projections[stage] = projection_optimizer
        self.ridge_functions[stage] = ridge_function

    def transform(self, x: np.ndarray):
        """Projects data to lower dimensions."""
        check_is_fitted(self, ["projections", "ridge_functions"])
        x = check_array(x)
        return np.array(x) @ np.concatenate(
            [projection.beta for projection in self.projections if projection is not None], axis=1,
        )

    def predict(self, x: np.ndarray):
        """Outputs inferences from data."""
        projected = self.transform(x)
        yhat = self.learning_rate * sum(
            (
                self.ridge_functions[stage].predict(projected[:, [stage]])
                for stage in range(self.n_stages)
            )
        )
        return yhat

    def plot_losses(self, ax: Optional[plt.Axes] = None):
        """Plots losses over each stage and iteration."""
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 5))

        iteration_losses = [loss for stage in self.loss_path for loss in stage]
        ax.plot(iteration_losses, marker=".", linestyle=":", markersize=3, linewidth=1)

        stage_indices = np.cumsum([len(stage) for stage in self.loss_path]) - 1
        for stage in stage_indices:
            ax.axvline(stage, color="red", linestyle="--", alpha=0.25)

        ax.set_xlabel("iteration count")
        ax.set_ylabel("loss")
        ax.set_title("loss across iterations")

    def plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        scatter_sample_ratio: float = 0.1,
        feature_names: List[str] = None,
        fig_height: float = 5.0,
        fig_width: float = 10.0,
    ) -> None:
        """Plots one-dimensional ridge functions and scatter plots."""
        if feature_names is None:
            feature_names = [f"x_{{{i}}}" for i in range(x.shape[1])]
        elif len(feature_names) != x.shape[1]:
            raise ValueError(
                f"Length of `feature_names` {len(feature_names)} differs from width of `x` {x.shape[1]}."
            )

        def _coef_parser(idx, beta, name):
            tokens = [f"{np.abs(beta):.3f}", "\\cdot", name]
            if beta >= 0 and idx != 0:
                tokens = ["+"] + tokens
            else:
                tokens = ["-"] + tokens

            return " ".join(tokens)

        fig, axs = plt.subplots(self.n_stages, 1, figsize=(fig_width, fig_height * self.n_stages))
        n_samples = x.shape[0]
        for stage in range(self.n_stages):
            xb = x @ self.projections[stage].beta
            xb_min = xb.min()
            xb_max = xb.max()
            xb_span = xb_max - xb_min

            # plot the ridge function
            xb_grid = np.linspace(
                start=xb_min - 0.05 * xb_span, stop=xb_max + 0.05 * xb_span, num=1000,
            )
            yhat_grid = self.ridge_functions[stage].predict(xb_grid.reshape(-1, 1))
            axs[stage].plot(
                xb_grid,
                yhat_grid,
                linewidth=2,
                color="r",
                label=f"ridge function {type(self.ridge_functions[stage]).__name__}",
            )

            # plot a subset of the data
            sample_indices = np.random.choice(
                n_samples, size=int(n_samples * scatter_sample_ratio), replace=False
            )
            axs[stage].plot(
                xb.ravel()[sample_indices],
                y.ravel()[sample_indices],
                marker=".",
                markersize=2,
                linestyle="",
                color="k",
                alpha=0.5,
                label="projected data",
            )

            # labeling
            projection_equation = " ".join(
                [
                    _coef_parser(idx=i, beta=b, name=feature_names[i])
                    for i, b in enumerate(self.projections[stage].beta.ravel())
                    if b != 0
                ]
            )
            axs[stage].set_title(
                f"Stage {stage}: $\\langle x, \\beta\\rangle = {projection_equation}$"
            )
            axs[stage].legend()

        fig.tight_layout()
