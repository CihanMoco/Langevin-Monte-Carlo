"""Friction sweep: reproduces Figure 5.

KLMC has an extra tuning knob that ULA/MALA don't: the friction
coefficient gamma. At fixed step size, sweep gamma across the
underdamped (gamma small - momentum-dominated, oscillatory) to
overdamped (gamma large - KLMC degenerates towards ULA) regimes and
measure mixing efficiency (min-coordinate ESS per gradient evaluation),
together with the stationary bias, on the same ill-conditioned Gaussian
target used in ``step_size_sweep.py``.

Momentum is what lets KLMC beat ULA's condition-number dependence
(Cheng et al., 2018; Dalalyan & Riou-Durand, 2020), and friction is what
throttles it: at fixed step size, more friction damps momentum faster,
so within the range where the frozen-force integrator stays numerically
reliable (see the note below), mixing efficiency decreases monotonically
as gamma grows - the chain smoothly interpolates from "momentum-assisted"
towards the ULA-like overdamped limit. Pushing gamma much lower still
would in principle help further, but the same frozen-force
discretisation that makes each step of KLMC cheap also becomes
inaccurate in the very-low-friction / near-Hamiltonian regime (its
deterministic part stops being a good, low-noise-amplification
integrator well before it becomes literally unstable) - which is itself
a useful, concrete illustration of the gap between an SDE's idealised
theory and what a fixed-cost discretisation can actually deliver.

Run with:  python experiments/friction_sweep.py
"""
from __future__ import annotations

import numpy as np

from _common import SAMPLER_STYLE, plt, save_figure, save_results  # noqa: E402
from src.diagnostics.mixing import min_ess
from src.samplers.klmc import klmc_sample
from src.targets.gaussian import make_ill_conditioned_gaussian

SEED = 1
DIM = 10
CONDITION_NUMBER = 15.0
N_STEPS = 40000
N_BURNIN = 5000
N_GAMMAS = 16
N_REPEATS = 5


def run_sweep():
    rng = np.random.default_rng(SEED)
    target = make_ill_conditioned_gaussian(DIM, CONDITION_NUMBER, rng=rng)
    L = target.smoothness

    # A fixed step size (well below the ULA-equivalent stability limit
    # 2/L) so the sweep isolates the effect of friction alone.
    #
    # Note on numerics: the "exact OU, frozen force" integrator used by
    # klmc_sample freezes grad_log_pi over each step. For this Gaussian
    # target the resulting deterministic recursion is *exactly* linear,
    # and one can check its spectral radius rho(gamma, h, lambda)
    # directly (see the derivation in klmc.py's docstring). rho < 1
    # everywhere friction is not vanishingly small, but as gamma -> 0
    # rho -> 1 and the stationary variance of a near-marginally-stable
    # linear recursion driven by noise blows up like 1/(1 - rho) - i.e.
    # the chain does not "diverge" outright, but its stationary
    # covariance becomes enormous long before rho actually crosses 1.
    # We therefore keep gamma comfortably clear of that regime.
    sqrtL = np.sqrt(L)
    h = 0.3 / L
    gammas = np.logspace(np.log10(0.3 * sqrtL), np.log10(30.0 * sqrtL), N_GAMMAS)
    x0 = np.zeros(DIM)

    ess_per_grad = []
    bias = []
    true_second_moment = np.trace(target.cov)

    for gamma in gammas:
        ess_vals = []
        bias_vals = []
        unstable = False
        for _ in range(N_REPEATS):
            seed_rng = np.random.default_rng(rng.integers(1 << 32))
            kres = klmc_sample(target.grad_log_prob, x0, h, N_STEPS, gamma=gamma, rng=seed_rng)
            post = kres.x_chain[N_BURNIN:]
            if not np.all(np.isfinite(post)) or np.abs(post).max() > 1e4:
                unstable = True
                break
            ess_vals.append(min_ess(post) / len(post))
            sq_dist = np.sum((post - target.mean) ** 2, axis=1)
            bias_vals.append((sq_dist.mean() - true_second_moment) / true_second_moment)

        if unstable:
            print(f"gamma={gamma:.3g} (gamma/sqrt(L)={gamma/sqrtL:.2f})  -- numerically unstable, skipped")
            ess_per_grad.append(float("nan"))
            bias.append(float("nan"))
            continue

        # Median across repeats: at low friction the chain decorrelates
        # slowly, so a single repeat's second-moment estimate can be a
        # noisy outlier - the median is far more robust here than the
        # mean of pooled samples.
        median_bias = float(np.median(bias_vals))
        if abs(median_bias) > 1.0:
            # Comfortably finite but enormous relative bias is the
            # signature of near-marginal stability (see module docstring
            # above), not a meaningful "high-bias" data point - flag it
            # rather than plot a misleading number.
            print(f"gamma={gamma:.3g} (gamma/sqrt(L)={gamma/sqrtL:.2f})  "
                  f"-- near-marginal stability (bias={median_bias:.3g}), skipped")
            ess_per_grad.append(float("nan"))
            bias.append(float("nan"))
            continue

        ess_per_grad.append(float(np.mean(ess_vals)))
        bias.append(median_bias)
        print(f"gamma={gamma:.3g} (gamma/sqrt(L)={gamma/sqrtL:.2f})  "
              f"ess/grad={ess_per_grad[-1]:.4g}  bias={bias[-1]:.4g}")

    results = {
        "gammas": gammas,
        "gamma_over_sqrtL": gammas / sqrtL,
        "step_size": h,
        "dim": DIM,
        "condition_number": CONDITION_NUMBER,
        "ess_per_grad": ess_per_grad,
        "bias": bias,
    }
    return gammas, results


def make_plot(gammas, results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(gammas, results["ess_per_grad"], **SAMPLER_STYLE["KLMC"])
    ax1.set_xscale("log")
    ax1.set_xlabel("friction gamma")
    ax1.set_ylabel("min-coordinate ESS / gradient evaluation")
    ax1.set_title("KLMC mixing efficiency vs friction")
    best = int(np.nanargmax(results["ess_per_grad"]))
    ax1.axvline(gammas[best], color="gray", linestyle="--", linewidth=1,
                label=f"best: gamma={gammas[best]:.2g}")
    ax1.legend()

    ax2.plot(gammas, results["bias"], **SAMPLER_STYLE["KLMC"])
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("friction gamma")
    ax2.set_ylabel(r"relative bias in $\mathrm{tr}(\mathrm{Cov})$")
    ax2.set_title("KLMC stationary bias vs friction")

    fig.suptitle("Figure 5: effect of friction on KLMC")
    fig.tight_layout()
    save_figure(fig, "fig5_friction_sweep")


if __name__ == "__main__":
    gammas, results = run_sweep()
    save_results("friction_sweep", results)
    make_plot(gammas, results)
