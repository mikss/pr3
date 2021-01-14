from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Set, Tuple, Union

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_random_state
from statsmodels.nonparametric.kernel_regression import KernelReg


class UnivariateNonlinearRegressor(ABC):
    _rs: np.random.RandomState

    def __init__(self, random_state: Optional[Union[int, np.random.RandomState]] = None):
        self._rs = check_random_state(random_state)

    def fit(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> None:
        """Fits a univariate regressor."""
        if w is None:
            w = np.ones(x.shape[0])
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

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Outputs derivatives of learned regression function"""

    def weighted_resampler(
        self, x: np.ndarray, y: np.ndarray, w: np.ndarray, bootstrap_ratio: float = 4.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = w.shape[0]
        indices = self._rs.choice(
            a=n_samples, size=int(bootstrap_ratio * n_samples), replace=True, p=w / w.sum(),
        )
        x = x[indices, :]
        y = y[indices, :]
        return x, y


class PiecewiseLinearUNLR(UnivariateNonlinearRegressor):
    perceptron: MLPRegressor
    scale: float

    def __init__(
        self,
        components: int = 25,
        max_iter: int = 2000,
        tol: float = 1e-4,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__(random_state)
        self.perceptron = MLPRegressor(
            hidden_layer_sizes=(components,),
            activation="relu",
            solver="lbfgs",
            random_state=self._rs,
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

    @staticmethod
    def _drelu(a: np.ndarray):
        return (a >= 0).astype(float)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative of MLP.

        Algebra: (where . means dot product, o means elementwise)
            n: samples
            p: feature dimension
            s: stage count
            x: (n, p) features
            y: (n, 1) response
            a: (1, s) intercepts
            b: (p, s) coefficients
            c: (1, 1) intercept
            d: (s, 1) coefficients

            y = c + relu(a + x @ b) @ d
            dy/dx = (relu'(a + x @ b) o b) @ d
        """
        mlp = self.perceptron
        a = mlp.intercepts_[0].reshape((1, -1))
        b = mlp.coefs_[0]
        d = mlp.coefs_[1]
        return (self._drelu(a + x @ b) * b) @ d


class PolynomialUNLR(UnivariateNonlinearRegressor):
    degree: int
    polynomial: np.poly1d

    def __init__(
        self, degree: int = 6, random_state: Optional[Union[int, np.random.RandomState]] = None
    ):
        super().__init__(random_state)
        self.degree = degree

    def _fit_univariate(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> None:
        self.polynomial = np.poly1d(
            np.polyfit(x=x.ravel(), y=y.ravel(), deg=self.degree, w=np.sqrt(w.ravel()),)
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.polynomial(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.polynomial.deriv()(x)


class NadarayaWatsonUNLR(UnivariateNonlinearRegressor):
    kernel: KernelReg
    bandwidth: float

    def __init__(
        self,
        bandwidth: float = 0.25,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__(random_state)
        self.bandwidth = bandwidth

    def _fit_univariate(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> None:
        if w is not None:
            x, y = self.weighted_resampler(x, y, w)
        self.kernel = KernelReg(endog=y, exog=x, var_type="c", bw=[self.bandwidth])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.kernel.fit(x)[0]

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.kernel.fit(x)[1]


class RidgeFunctionRegistry(Enum):
    piecewise = PiecewiseLinearUNLR
    polynomial = PolynomialUNLR
    kernel = NadarayaWatsonUNLR

    @classmethod
    def valid_mnemonics(cls) -> Set[str]:
        return set(name for name, _ in cls.__members__.items())

    @classmethod
    def valid_regressors(cls) -> Set[UnivariateNonlinearRegressor]:
        return set(value for _, value in cls.__members__.items())
