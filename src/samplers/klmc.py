"""Kinetic (underdamped) Langevin Monte Carlo (KLMC).

KLMC targets pi(x) by simulating the *underdamped* Langevin diffusion on
the augmented state (position x, velocity v):

    dV_t = -gamma * V_t dt + u * grad_log_pi(X_t) dt + sqrt(2 * gamma * u) dB_t
    dX_t = V_t dt

where ``gamma`` is the friction coefficient and ``u = 1 / mass`` is the
inverse mass. Marginalising out V, X has pi as its stationary law.

Discretisation
---------------
Rather than a naive Euler-Maruyama step, we use the "exact" integrator
that freezes the force ``grad_log_pi`` at the start of each step and
integrates the resulting *linear* (Ornstein-Uhlenbeck) system in closed
form. This is the discretisation analysed in

    Cheng, Chatterji, Bartlett & Jordan (2018), "Underdamped Langevin
    MCMC: A non-asymptotic analysis", COLT.

    Dalalyan & Riou-Durand (2020), "On sampling from a log-concave
    density using kinetic Langevin diffusions", Bernoulli.

and gives, conditional on (x_k, v_k) and freezing g = grad_log_pi(x_k)
over [0, h]:

    psi(h)   = (1 - exp(-gamma h)) / gamma
    mean_v   = exp(-gamma h) v_k + u psi(h) g
    mean_x   = x_k + psi(h) v_k + (u / gamma) (h - psi(h)) g

with (x_{k+1} - mean_x, v_{k+1} - mean_v) jointly Gaussian, independent
across coordinates, with per-coordinate covariance

    Var(v)   = u (1 - exp(-2 gamma h))
    Var(x)   = (2u / gamma) [ h - 2 psi(h) + (1 - exp(-2 gamma h)) / (2 gamma) ]
    Cov(x,v) = 2u [ psi(h) - (1 - exp(-2 gamma h)) / (2 gamma) ]

which is obtained by solving the linear SDE for V in closed form and
integrating X_t = x_k + int_0^t V_s ds exactly.

This reduces, in the high-friction / small-step limit, to ULA; the
friction gamma trades off how quickly momentum decorrelates against how
much the discretisation noise is damped, which is exactly what
``experiments/friction_sweep.py`` explores.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import numpy as np


class KLMCResult(NamedTuple):
    x_chain: np.ndarray  # (n_steps + 1, d) position trajectory
    v_chain: np.ndarray  # (n_steps + 1, d) velocity trajectory


def _ou_moments(gamma: float, u: float, h: float):
    """Closed-form moments of the frozen-force OU integrator over step h."""
    e1 = np.exp(-gamma * h)
    e2 = e1 * e1  # exp(-2 gamma h)
    psi = (1.0 - e1) / gamma

    var_v = u * (1.0 - e2)
    var_x = (2.0 * u / gamma) * (h - 2.0 * psi + (1.0 - e2) / (2.0 * gamma))
    cov_xv = 2.0 * u * (psi - (1.0 - e2) / (2.0 * gamma))

    # Numerical floor: for very small gamma*h these can dip slightly
    # negative due to cancellation error; clip to keep Cholesky well-defined.
    var_v = max(var_v, 0.0)
    var_x = max(var_x, 0.0)

    return e1, psi, var_x, var_v, cov_xv


def klmc_sample(
    grad_log_prob: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    step_size: float,
    n_steps: int,
    gamma: float = 2.0,
    u: float = 1.0,
    v0: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> KLMCResult:
    """Run Kinetic Langevin Monte Carlo with the exact OU integrator.

    Parameters
    ----------
    grad_log_prob : callable
        Function mapping x (shape (d,)) to grad log pi(x) (shape (d,)).
    x0 : array_like, shape (d,)
        Initial position.
    step_size : float
        Integrator step size h > 0.
    n_steps : int
        Number of steps to run.
    gamma : float
        Friction coefficient (> 0). Large gamma => overdamped (ULA-like)
        regime; small gamma => underdamped, momentum-dominated regime.
    u : float
        Inverse mass, u = 1 / mass. Defaults to 1.0.
    v0 : array_like, shape (d,), optional
        Initial velocity. Defaults to a draw from the velocity marginal
        N(0, u * I), its stationary distribution.
    rng : numpy.random.Generator, optional
        Source of randomness.

    Returns
    -------
    KLMCResult
        ``x_chain`` and ``v_chain``, each of shape (n_steps + 1, d).
    """
    if rng is None:
        rng = np.random.default_rng()

    x0 = np.asarray(x0, dtype=float)
    d = x0.shape[0]
    h = float(step_size)
    if h <= 0:
        raise ValueError("step_size must be positive")
    if gamma <= 0:
        raise ValueError("gamma (friction) must be positive")

    if v0 is None:
        v0 = np.sqrt(u) * rng.standard_normal(d)
    else:
        v0 = np.asarray(v0, dtype=float)

    e1, psi, var_x, var_v, cov_xv = _ou_moments(gamma, u, h)

    # Cholesky of the shared 2x2 per-coordinate noise covariance.
    cov = np.array([[var_x, cov_xv], [cov_xv, var_v]])
    # Guard against tiny negative eigenvalues from floating point error.
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < -1e-10:
        raise RuntimeError("KLMC noise covariance is not PSD; check gamma/u/h")
    cov = cov + max(0.0, -eigvals.min() + 1e-12) * np.eye(2)
    L = np.linalg.cholesky(cov)

    x_chain = np.empty((n_steps + 1, d), dtype=float)
    v_chain = np.empty((n_steps + 1, d), dtype=float)
    x_chain[0] = x0
    v_chain[0] = v0

    x, v = x0.copy(), v0.copy()
    a = u * psi
    b = (u / gamma) * (h - psi)

    for k in range(n_steps):
        g = grad_log_prob(x)

        mean_x = x + psi * v + b * g
        mean_v = e1 * v + a * g

        z = rng.standard_normal((d, 2))
        noise = z @ L.T  # shape (d, 2): columns are (noise_x, noise_v)

        x = mean_x + noise[:, 0]
        v = mean_v + noise[:, 1]

        x_chain[k + 1] = x
        v_chain[k + 1] = v

    return KLMCResult(x_chain=x_chain, v_chain=v_chain)


class KLMC:
    """Thin object-oriented wrapper around :func:`klmc_sample`."""

    def __init__(self, target, step_size: float, gamma: float = 2.0, u: float = 1.0):
        self.target = target
        self.step_size = step_size
        self.gamma = gamma
        self.u = u

    def run(
        self,
        x0: np.ndarray,
        n_steps: int,
        v0: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> KLMCResult:
        return klmc_sample(
            self.target.grad_log_prob,
            x0,
            self.step_size,
            n_steps,
            gamma=self.gamma,
            u=self.u,
            v0=v0,
            rng=rng,
        )
