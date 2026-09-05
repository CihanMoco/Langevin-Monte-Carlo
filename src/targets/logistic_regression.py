"""Bayesian logistic regression target.

Model:
    w          ~ N(0, tau2 * I_d)                         (prior)
    y_i | x_i  ~ Bernoulli(sigmoid(x_i^T w)),  i = 1..n    (likelihood)

Unlike the Gaussian and linear-regression targets, the logistic
regression posterior is not available in closed form, but it is still
strongly log-concave (for tau2 < infinity), so ULA/MALA/KLMC all have
non-asymptotic guarantees here. The log-density and its gradient are
computed in a numerically stable way via the log-sum-exp trick.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _log_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable log(sigmoid(z)) = -log(1 + exp(-z))."""
    return -np.logaddexp(0.0, -z)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class BayesianLogisticRegression:
    """Log-posterior and gradient for Bayesian logistic regression.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Design matrix (include a column of ones for an intercept if
        desired).
    y : ndarray, shape (n,)
        Binary labels in {0, 1}.
    tau2 : float
        Prior variance on each coefficient of w.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, tau2: float = 1.0):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.tau2 = float(tau2)
        self.n, self.dim = self.X.shape

    def log_prob(self, w: np.ndarray) -> float:
        z = self.X @ w
        log_lik = np.sum(self.y * _log_sigmoid(z) + (1.0 - self.y) * _log_sigmoid(-z))
        log_prior = -0.5 / self.tau2 * np.dot(w, w)
        return log_lik + log_prior

    def grad_log_prob(self, w: np.ndarray) -> np.ndarray:
        z = self.X @ w
        p = _sigmoid(z)
        grad_lik = self.X.T @ (self.y - p)
        grad_prior = -w / self.tau2
        return grad_lik + grad_prior

    @classmethod
    def generate_data(
        cls,
        n: int,
        dim: int,
        tau2: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> "BayesianLogisticRegression":
        """Simulate a design matrix, ground-truth weights ~ prior, and
        Bernoulli labels, and return the resulting posterior target."""
        if rng is None:
            rng = np.random.default_rng()
        X = rng.standard_normal((n, dim))
        w_true = np.sqrt(tau2) * rng.standard_normal(dim)
        p = _sigmoid(X @ w_true)
        y = (rng.uniform(size=n) < p).astype(float)
        model = cls(X, y, tau2=tau2)
        model.w_true = w_true
        return model
