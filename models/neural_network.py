import numpy as np
from sklearn.metrics import r2_score


class NeuralNetworkModel:
    """Simple feedforward neural network trained with backpropagation (numpy only)"""

    def __init__(self, hidden=8, lr=0.01, epochs=5000):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.trained = False

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def _forward(self, X):
        h = self._sigmoid(X @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out, h

    def fit(self, X, y):
        np.random.seed(42)
        n_in = X.shape[1]
        self.W1 = np.random.randn(n_in, self.hidden) * 0.5
        self.b1 = np.zeros(self.hidden)
        self.W2 = np.random.randn(self.hidden, 1) * 0.5
        self.b2 = np.zeros(1)

        self.X_mean = X.mean(axis=0)
        self.X_std  = X.std(axis=0) + 1e-8
        self.y_mean = y.mean()
        self.y_std  = y.std() + 1e-8
        Xn = (X - self.X_mean) / self.X_std
        yn = (y - self.y_mean) / self.y_std

        for _ in range(self.epochs):
            out, h = self._forward(Xn)
            loss = out[:, 0] - yn
            dW2 = h.T @ loss[:, None] / len(y)
            db2 = loss.mean()
            dh  = (loss[:, None] @ self.W2.T) * h * (1 - h)
            dW1 = Xn.T @ dh / len(y)
            db1 = dh.mean(axis=0)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

        self.trained = True
        return self

    def predict(self, X):
        Xn = (X - self.X_mean) / self.X_std
        out, _ = self._forward(Xn)
        return out[:, 0] * self.y_std + self.y_mean

    def r2(self, X, y):
        return r2_score(y, self.predict(X))
