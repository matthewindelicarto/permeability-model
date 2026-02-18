import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score


class GaussianProcessModel:
    """Gaussian Process regression with Matern kernel — ideal for small scientific datasets"""

    def __init__(self):
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
            Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) +
            WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=42,
        )
        self.trained = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.trained = True
        return self

    def predict(self, X):
        return self.model.predict(np.atleast_2d(X))

    def r2(self, X, y):
        return r2_score(y, self.predict(X))
