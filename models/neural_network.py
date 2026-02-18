import numpy as np
from sklearn.metrics import r2_score


class NeuralNetworkModel:
    """Ensemble of small feedforward NNs trained with backpropagation (numpy only).
    Uses L2 regularization and averages over multiple random seeds to reduce
    overfitting on small datasets."""

    def __init__(self, hidden=4, lr=0.01, epochs=5000, weight_decay=1e-3, n_ensemble=5):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.n_ensemble = n_ensemble
        self.trained = False

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def _forward(self, Xn, W1, b1, W2, b2):
        h = self._sigmoid(Xn @ W1 + b1)
        out = h @ W2 + b2
        return out, h

    def _train_one(self, Xn, yn, seed):
        np.random.seed(seed)
        n_in = Xn.shape[1]
        W1 = np.random.randn(n_in, self.hidden) * 0.5
        b1 = np.zeros(self.hidden)
        W2 = np.random.randn(self.hidden, 1) * 0.5
        b2 = np.zeros(1)

        for _ in range(self.epochs):
            out, h = self._forward(Xn, W1, b1, W2, b2)
            loss = out[:, 0] - yn
            dW2 = h.T @ loss[:, None] / len(yn)
            db2 = loss.mean()
            dh  = (loss[:, None] @ W2.T) * h * (1 - h)
            dW1 = Xn.T @ dh / len(yn)
            db1 = dh.mean(axis=0)
            W2 -= self.lr * (dW2 + self.weight_decay * W2)
            b2 -= self.lr * db2
            W1 -= self.lr * (dW1 + self.weight_decay * W1)
            b1 -= self.lr * db1

        return W1, b1, W2, b2

    def fit(self, X, y):
        self.X_mean = X.mean(axis=0)
        self.X_std  = X.std(axis=0) + 1e-8
        self.y_mean = y.mean()
        self.y_std  = y.std() + 1e-8
        Xn = (X - self.X_mean) / self.X_std
        yn = (y - self.y_mean) / self.y_std

        self.ensemble = [
            self._train_one(Xn, yn, seed=i) for i in range(self.n_ensemble)
        ]
        self.trained = True
        return self

    def predict(self, X):
        Xn = (np.atleast_2d(X) - self.X_mean) / self.X_std
        preds = []
        for W1, b1, W2, b2 in self.ensemble:
            out, _ = self._forward(Xn, W1, b1, W2, b2)
            preds.append(out[:, 0])
        avg = np.mean(preds, axis=0)
        return avg * self.y_std + self.y_mean

    def r2(self, X, y):
        return r2_score(y, self.predict(X))
