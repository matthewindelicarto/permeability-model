import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


class RegressionModel:
    """Polynomial ridge regression model (degree 2)"""

    def __init__(self):
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=True)),
            ("ridge", Ridge(alpha=1.0))
        ])
        self.trained = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.trained = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def r2(self, X, y):
        return r2_score(y, self.predict(X))
