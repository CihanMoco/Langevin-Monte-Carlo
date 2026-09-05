"""Correlated Gaussian benchmark target.

The Gaussian target is the standard sanity-check and stress-test for
Langevin samplers: the posterior mean/covariance are known in closed
form, so sampler error can be measured exactly, and its condition number
kappa = L / m (ratio of the largest to smallest eigenvalue of the
precision matrix, i.e. of the log-density Hessian) can be dialled up to
probe the theoretical dependence of mixing time on conditioning
(see ``experiments/precision_scaling.py``) and on dimension
(see ``experiments/dimension_scaling.py``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class CorrelatedGaussian:
    """N(mean, cov) target with log_prob / grad_log_prob for LMC samplers.

    Parameters
    ----------
    mean : array_like, shape (d,)
    cov : array_like, shape (d, d)
        Covariance matrix (must be symmetric positive definite).
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.dim = self.mean.shape[0]

        # Precompute precision matrix (the Hessian of -log pi) and a
        # Cholesky factor of the covariance for exact ground-truth sampling.
        self.precision = np.linalg.inv(self.cov)
        self._chol_cov = np.linalg.cholesky(self.cov)
        sign, logdet = np.linalg.slogdet(self.cov)
        self._log_norm_const = -0.5 * (self.dim * np.log(2 * np.pi) + logdet)

        eigvals = np.linalg.eigvalsh(self.precision)
        self.smoothness = eigvals.max()  # L: largest eigenvalue of the Hessian
        self.strong_convexity = eigvals.min()  # m: smallest eigenvalue
        self.condition_number = self.smoothness / self.strong_convexity

    def log_prob(self, x: np.ndarray) -> float:
        diff = x - self.mean
        quad = diff @ self.precision @ diff
        return self._log_norm_const - 0.5 * quad

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        return -self.precision @ (x - self.mean)

    def exact_sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw n i.i.d. exact samples (for ground-truth comparisons)."""
        if rng is None:
            rng = np.random.default_rng()
        z = rng.standard_normal((n, self.dim))
        return self.mean + z @ self._chol_cov.T


def make_ill_conditioned_gaussian(
    dim: int,
    condition_number: float,
    rng: Optional[np.random.Generator] = None,
    mean: Optional[np.ndarray] = None,
) -> CorrelatedGaussian:
    """Build a zero-mean (by default) correlated Gaussian whose precision
    matrix has a prescribed condition number.

    The precision matrix is constructed as Lambda = Q diag(eigs) Q^T with
    Q a Haar-random orthogonal matrix and eigs log-spaced between
    1/sqrt(condition_number) and sqrt(condition_number) (so that the
    ratio of largest to smallest eigenvalue is exactly
    ``condition_number``). This is the standard construction used to
    probe how LMC mixing degrades with ill-conditioning.
    """
    if rng is None:
        rng = np.random.default_rng()
    if dim == 1:
        precision = np.array([[1.0]])
    else:
        # Haar-random orthogonal matrix via QR decomposition of a
        # standard Gaussian matrix (with a sign fix for uniformity).
        a = rng.standard_normal((dim, dim))
        q, r = np.linalg.qr(a)
        q = q * np.sign(np.diag(r))

        eigs = np.logspace(
            -0.5 * np.log10(condition_number),
            0.5 * np.log10(condition_number),
            dim,
        )
        precision = (q * eigs) @ q.T
        precision = 0.5 * (precision + precision.T)  # symmetrise (numerical safety)

    cov = np.linalg.inv(precision)
    if mean is None:
        mean = np.zeros(dim)
    return CorrelatedGaussian(mean, cov)
