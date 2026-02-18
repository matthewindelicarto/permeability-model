import numpy as np
from sklearn.metrics import r2_score


class NeuralNetworkModel:
    """Ensemble of feedforward NNs trained with Adam optimizer (numpy only).
    Ensemble averaging over multiple seeds reduces seed-dependent artifacts.
    Adam handles tight output ranges and sparse gradients far better than SGD."""

    def __init__(self, hidden=6, epochs=8000, weight_decay=1e-3, n_ensemble=7,
                 lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.hidden = hidden
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.n_ensemble = n_ensemble
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
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
        # He initialization — better for sigmoid activations on small datasets
        W1 = np.random.randn(n_in, self.hidden) * np.sqrt(2.0 / n_in)
        b1 = np.zeros(self.hidden)
        W2 = np.random.randn(self.hidden, 1) * np.sqrt(2.0 / self.hidden)
        b2 = np.zeros(1)

        # Adam moment accumulators
        mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
        mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
        mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
        mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)

        for t in range(1, self.epochs + 1):
            out, h = self._forward(Xn, W1, b1, W2, b2)
            loss = out[:, 0] - yn

            gW2 = h.T @ loss[:, None] / len(yn) + self.weight_decay * W2
            gb2 = np.array([loss.mean()])
            dh  = (loss[:, None] @ W2.T) * h * (1 - h)
            gW1 = Xn.T @ dh / len(yn) + self.weight_decay * W1
            gb1 = dh.mean(axis=0)

            bc1 = 1 - self.beta1 ** t
            bc2 = 1 - self.beta2 ** t

            def adam_step(p, g, m, v):
                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * g ** 2
                p -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)
                return p, m, v

            W1, mW1, vW1 = adam_step(W1, gW1, mW1, vW1)
            b1, mb1, vb1 = adam_step(b1, gb1, mb1, vb1)
            W2, mW2, vW2 = adam_step(W2, gW2, mW2, vW2)
            b2, mb2, vb2 = adam_step(b2, gb2, mb2, vb2)

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
