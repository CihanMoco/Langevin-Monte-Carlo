"""Metropolis-Adjusted Langevin Algorithm (MALA).

MALA uses the same Langevin proposal as ULA,

    y = x + h * grad_log_pi(x) + sqrt(2h) * xi,    xi ~ N(0, I),

but corrects the discretisation bias with a Metropolis-Hastings accept /
reject step, so the chain has pi as its exact stationary distribution
(assuming it is otherwise ergodic). The trade-off is a per-iteration
acceptance probability that can be small if h is too large, and the need
to evaluate the (possibly expensive) target log-density at every step.

Acceptance probability
-----------------------
    alpha(x, y) = min(1, [pi(y) q(x|y)] / [pi(x) q(y|x)])

where q(y|x) = N(y; x + h grad_log_pi(x), 2h I) is the Langevin proposal
density. Because q(y|x) and q(x|y) share the same (constant) normalising
constant, only the quadratic forms in the exponent are needed.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import numpy as np


class MALAResult(NamedTuple):
    chain: np.ndarray
    accept_rate: float
    accepted: np.ndarray  # boolean array, one entry per proposed move


def _log_proposal_quadratic(y: np.ndarray, x: np.ndarray, grad_x: np.ndarray, h: float) -> float:
    """Return -||y - x - h*grad_x||^2 / (4h), the log proposal density of
    y given x up to the (shared, cancelling) normalising constant."""
    diff = y - x - h * grad_x
    return -np.dot(diff, diff) / (4.0 * h)


def mala_sample(
    log_prob: Callable[[np.ndarray], float],
    grad_log_prob: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    step_size: float,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
) -> MALAResult:
    """Run the Metropolis-Adjusted Langevin Algorithm.

    Parameters
    ----------
    log_prob : callable
        Function mapping x (shape (d,)) to log pi(x) (up to a constant).
    grad_log_prob : callable
        Function mapping x (shape (d,)) to grad log pi(x) (shape (d,)).
    x0 : array_like, shape (d,)
        Initial state of the chain.
    step_size : float
        Langevin proposal step size h > 0.
    n_steps : int
        Number of MH iterations to run (the returned chain has n_steps + 1
        rows, including x0).
    rng : numpy.random.Generator, optional
        Source of randomness. A fresh default_rng() is used if omitted.

    Returns
    -------
    MALAResult
        ``chain`` (n_steps + 1, d), ``accept_rate`` (float in [0, 1]) and
        ``accepted`` (boolean array of length n_steps).
    """
    if rng is None:
        rng = np.random.default_rng()

    x0 = np.asarray(x0, dtype=float)
    d = x0.shape[0]
    h = float(step_size)
    if h <= 0:
        raise ValueError("step_size must be positive")

    chain = np.empty((n_steps + 1, d), dtype=float)
    chain[0] = x0
    accepted = np.zeros(n_steps, dtype=bool)

    x = x0.copy()
    log_pi_x = log_prob(x)
    grad_x = grad_log_prob(x)
    scale = np.sqrt(2.0 * h)

    n_accept = 0
    for k in range(n_steps):
        y = x + h * grad_x + scale * rng.standard_normal(d)
        log_pi_y = log_prob(y)
        grad_y = grad_log_prob(y)

        log_q_forward = _log_proposal_quadratic(y, x, grad_x, h)
        log_q_backward = _log_proposal_quadratic(x, y, grad_y, h)

        log_alpha = (log_pi_y - log_pi_x) + (log_q_backward - log_q_forward)

        if np.log(rng.uniform()) < min(0.0, log_alpha):
            x, log_pi_x, grad_x = y, log_pi_y, grad_y
            accepted[k] = True
            n_accept += 1

        chain[k + 1] = x

    return MALAResult(chain=chain, accept_rate=n_accept / n_steps, accepted=accepted)


class MALA:
    """Thin object-oriented wrapper around :func:`mala_sample`."""

    def __init__(self, target, step_size: float):
        self.target = target
        self.step_size = step_size

    def run(self, x0: np.ndarray, n_steps: int, rng: Optional[np.random.Generator] = None) -> MALAResult:
        return mala_sample(
            self.target.log_prob,
            self.target.grad_log_prob,
            x0,
            self.step_size,
            n_steps,
            rng=rng,
        )
