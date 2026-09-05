"""Unadjusted Langevin Algorithm (ULA).

ULA discretises the overdamped Langevin diffusion

    dX_t = grad_log_pi(X_t) dt + sqrt(2) dB_t,

whose stationary distribution is the target density pi, via a simple
Euler-Maruyama scheme:

    x_{k+1} = x_k + h * grad_log_pi(x_k) + sqrt(2h) * xi_k,    xi_k ~ N(0, I)

Because the Euler-Maruyama step introduces a discretisation bias, ULA does
not target pi exactly for h > 0: it converges to a biased stationary
distribution whose distance from pi shrinks with the step size h. It also
never rejects, so it is much cheaper per iteration than MALA.

References
----------
Roberts, G. O. and Tweedie, R. L. (1996). Exponential convergence of
Langevin distributions and their discrete approximations.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def ula_sample(
    grad_log_prob: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    step_size: float,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Run the Unadjusted Langevin Algorithm.

    Parameters
    ----------
    grad_log_prob : callable
        Function mapping x (shape (d,)) to grad log pi(x) (shape (d,)).
    x0 : array_like, shape (d,)
        Initial state of the chain.
    step_size : float
        Euler-Maruyama step size h > 0.
    n_steps : int
        Number of iterations to run (the returned chain has n_steps + 1
        rows, including x0).
    rng : numpy.random.Generator, optional
        Source of randomness. A fresh default_rng() is used if omitted.

    Returns
    -------
    chain : ndarray, shape (n_steps + 1, d)
        The sampled trajectory, chain[0] == x0.
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

    x = x0.copy()
    scale = np.sqrt(2.0 * h)
    for k in range(n_steps):
        grad = grad_log_prob(x)
        x = x + h * grad + scale * rng.standard_normal(d)
        chain[k + 1] = x

    return chain


class ULA:
    """Thin object-oriented wrapper around :func:`ula_sample`.

    Lets callers write ``ULA(target, step_size).run(x0, n_steps)`` which is
    convenient when sweeping over many targets/step sizes in the
    experiment scripts.
    """

    def __init__(self, target, step_size: float):
        self.target = target
        self.step_size = step_size

    def run(self, x0: np.ndarray, n_steps: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return ula_sample(self.target.grad_log_prob, x0, self.step_size, n_steps, rng=rng)
