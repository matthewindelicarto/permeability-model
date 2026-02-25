import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.metrics import r2_score


class GaussianProcessModel:
    """Gaussian Process regression with Matern kernel — ideal for small scientific datasets.

    The `alpha` parameter fixes the noise variance on the diagonal of the kernel matrix.
    Rather than letting the GP optimise noise freely (WhiteKernel), we set it from
    experimental replicates: α = mean(Δ²/2) across repeated membrane tests, where Δ
    is the difference in log₁₀(P) between two runs of the same composition.
    """

    def __init__(self, alpha=1e-6):
        """
        Parameters
        ----------
        alpha : float
            Noise variance in log₁₀(P) units, estimated from experimental replicates.
            Defaults to near-zero (essentially noiseless) when no replicates are available.
        """
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
            Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
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
