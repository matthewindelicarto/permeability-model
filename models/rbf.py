import numpy as np
from sklearn.metrics import r2_score


class RBFModel:
    """Gaussian radial basis function interpolation model"""

    def __init__(self):
        self.trained = False

    def _gaussian(self, r, eps):
        return np.exp(-(eps * r) ** 2)

    def _dist_matrix(self, X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        D = self._dist_matrix(X, X)
        np.fill_diagonal(D, np.nan)
        med = np.nanmedian(D)
        self.eps = 1.0 / (med + 1e-8)
        Phi = self._gaussian(self._dist_matrix(X, X), self.eps)
        Phi += np.eye(len(y)) * 1e-6
        self.weights = np.linalg.solve(Phi, y)
        self.trained = True
        return self

    def predict(self, X):
        D = self._dist_matrix(np.atleast_2d(X), self.X_train)
        Phi = self._gaussian(D, self.eps)
        return Phi @ self.weights

    def r2(self, X, y):
        return r2_score(y, self.predict(X))
