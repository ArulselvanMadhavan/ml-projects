"""Pure Python."""
import numpy as np


class LinearRegressionGD(object):
    """Linear Regression using Grad Descent."""

    def __init__(self, eta=0.001, n_iter=20):
        """Constructor."""
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit."""
        self._w = np.zeros(X.shape[1] + 1)
        self._cost = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self._w[1:] += self.eta * X.T.dot(errors)
            self._w[0] += self.eta * errors.sum()
            cost_iter = (errors ** 2).sum() / 2.0
            self._cost.append(cost_iter)
        return self

    def net_input(self, X):
        """Compute the dot product between X and W.

        X - NxD
        w - 1xD
        """
        return np.dot(X, self._w[1:]) + self._w[0]

    def predict(self, X):
        """Predict."""
        return self.net_input(X)
