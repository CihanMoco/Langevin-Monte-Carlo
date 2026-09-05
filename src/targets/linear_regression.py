"""Bayesian linear regression target.

Model:
    w        ~ N(0, tau2 * I_d)                (prior)
    y | X, w ~ N(X w, sigma2 * I_n)             (likelihood)

The posterior over w is available in closed form (Gaussian-Gaussian
conjugacy), which makes this a convenient, still-nontrivial benchmark:
LMC samplers can be checked against the exact posterior mean/covariance,
while the log-density's Hessian (X^T X / sigma2 + I / tau2) is generally
ill-conditioned when the design matrix X is, unlike the hand-built
CorrelatedGaussian target.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class BayesianLinearRegression:
    """Log-posterior and gradient for Bayesian linear regression.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Design matrix.
    y : ndarray, shape (n,)
        Response vector.
    sigma2 : float
        Observation noise variance.
    tau2 : float
        Prior variance on each coefficient of w.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, sigma2: float = 1.0, tau2: float = 1.0):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.sigma2 = float(sigma2)
        self.tau2 = float(tau2)
        self.n, self.dim = self.X.shape

        self._XtX = self.X.T @ self.X
        self._Xty = self.X.T @ self.y

    def log_prob(self, w: np.ndarray) -> float:
        resid = self.y - self.X @ w
        log_lik = -0.5 / self.sigma2 * np.dot(resid, resid)
        log_prior = -0.5 / self.tau2 * np.dot(w, w)
        return log_lik + log_prior

    def grad_log_prob(self, w: np.ndarray) -> np.ndarray:
        grad_lik = (self._Xty - self._XtX @ w) / self.sigma2
        grad_prior = -w / self.tau2
        return grad_lik + grad_prior

    def posterior_mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        """Exact Gaussian posterior N(mean, cov), for ground-truth checks."""
        precision = self._XtX / self.sigma2 + np.eye(self.dim) / self.tau2
        cov = np.linalg.inv(precision)
        mean = cov @ (self._Xty / self.sigma2)
        return mean, cov

    @property
    def condition_number(self) -> float:
        precision = self._XtX / self.sigma2 + np.eye(self.dim) / self.tau2
        eigvals = np.linalg.eigvalsh(precision)
        return eigvals.max() / eigvals.min()

    @classmethod
    def generate_data(
        cls,
        n: int,
        dim: int,
        sigma2: float = 1.0,
        tau2: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> "BayesianLinearRegression":
        """Simulate a design matrix, ground-truth weights ~ prior, and
        noisy responses, and return the resulting posterior target."""
        if rng is None:
            rng = np.random.default_rng()
        X = rng.standard_normal((n, dim))
        w_true = np.sqrt(tau2) * rng.standard_normal(dim)
        y = X @ w_true + np.sqrt(sigma2) * rng.standard_normal(n)
        model = cls(X, y, sigma2=sigma2, tau2=tau2)
        model.w_true = w_true
        return model
