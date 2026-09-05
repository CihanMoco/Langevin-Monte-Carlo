"""Step-size sweep: reproduces Figures 2-4.

For a fixed, moderately ill-conditioned correlated Gaussian target, sweep
the discretisation step size h and compare ULA, MALA and KLMC on:

    Figure 2 - bias:        relative bias in the second moment,
               (E_empirical[||X - mu*||^2] - trace(Sigma*)) / trace(Sigma*),
               vs h. For a Gaussian target the drift is linear, so the
               Euler-Maruyama recursion underlying ULA is an exact AR(1)
               process whose stationary *mean* equals the true mean for
               any stable h - the O(h) discretisation bias only shows up
               in the *covariance* (ULA systematically inflates it as h
               grows). MALA is asymptotically unbiased at every h thanks
               to the MH correction; KLMC's bias grows with h too, from
               its own frozen-gradient discretisation error.
    Figure 3 - mixing efficiency: (min per-coordinate ESS) / (number of
               gradient evaluations) vs h, for all three samplers.
    Figure 4 - MALA acceptance rate vs h (the classic accept-rate /
               step-size trade-off: bigger steps explore faster per
               accepted move but get rejected more often).

Run with:  python experiments/step_size_sweep.py
"""
from __future__ import annotations

import numpy as np

from _common import SAMPLER_STYLE, plt, save_figure, save_results  # noqa: E402
from src.diagnostics.mixing import min_ess
from src.samplers.klmc import klmc_sample
from src.samplers.mala import mala_sample
from src.samplers.ula import ula_sample
from src.targets.gaussian import make_ill_conditioned_gaussian

SEED = 0
DIM = 10
CONDITION_NUMBER = 15.0
N_STEPS = 40000
N_BURNIN = 5000
N_STEP_SIZES = 12
N_REPEATS = 3  # independent chains per step size, pooled to reduce Monte Carlo noise


def run_sweep():
    rng = np.random.default_rng(SEED)
    target = make_ill_conditioned_gaussian(DIM, CONDITION_NUMBER, rng=rng)
    L = target.smoothness  # Lipschitz constant of grad log pi (Hessian's top eigenvalue)
    gamma_klmc = 2.0 * np.sqrt(L)  # near-critical damping, see friction_sweep.py

    step_sizes = np.logspace(np.log10(0.02 / L), np.log10(1.5 / L), N_STEP_SIZES)
    x0 = np.zeros(DIM)

    results = {"step_sizes": step_sizes, "dim": DIM, "condition_number": CONDITION_NUMBER}
    for name in ("ULA", "MALA", "KLMC"):
        results[name] = {"bias": [], "ess_per_grad": []}
    results["MALA"]["accept_rate"] = []

    for h in step_sizes:
        pooled = {"ULA": [], "MALA": [], "KLMC": []}
        ess_per_grad = {"ULA": [], "MALA": [], "KLMC": []}
        accept_rates = []

        for _ in range(N_REPEATS):
            seed_rng = np.random.default_rng(rng.integers(1 << 32))

            chain = ula_sample(target.grad_log_prob, x0, h, N_STEPS, rng=seed_rng)
            post = chain[N_BURNIN:]
            pooled["ULA"].append(post)
            ess_per_grad["ULA"].append(min_ess(post) / len(post))

            mres = mala_sample(target.log_prob, target.grad_log_prob, x0, h, N_STEPS, rng=seed_rng)
            post = mres.chain[N_BURNIN:]
            pooled["MALA"].append(post)
            ess_per_grad["MALA"].append(min_ess(post) / len(post))
            accept_rates.append(mres.accept_rate)

            kres = klmc_sample(target.grad_log_prob, x0, h, N_STEPS, gamma=gamma_klmc, rng=seed_rng)
            post = kres.x_chain[N_BURNIN:]
            pooled["KLMC"].append(post)
            ess_per_grad["KLMC"].append(min_ess(post) / len(post))

        true_second_moment = np.trace(target.cov)
        for name in ("ULA", "MALA", "KLMC"):
            all_samples = np.concatenate(pooled[name], axis=0)
            sq_dist = np.sum((all_samples - target.mean) ** 2, axis=1)
            rel_bias = (sq_dist.mean() - true_second_moment) / true_second_moment
            results[name]["bias"].append(float(rel_bias))
            results[name]["ess_per_grad"].append(float(np.mean(ess_per_grad[name])))
        results["MALA"]["accept_rate"].append(float(np.mean(accept_rates)))

        print(f"h={h:.4g}  ULA bias={results['ULA']['bias'][-1]:.4g}  "
              f"MALA bias={results['MALA']['bias'][-1]:.4g} (acc={results['MALA']['accept_rate'][-1]:.2f})  "
              f"KLMC bias={results['KLMC']['bias'][-1]:.4g}")

    return step_sizes, results


def make_plots(step_sizes, results):
    # Figure 2: relative second-moment bias vs step size
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in ("ULA", "MALA", "KLMC"):
        ax.plot(step_sizes, results[name]["bias"], label=name, **SAMPLER_STYLE[name])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("step size h")
    ax.set_ylabel(r"relative bias in $\mathrm{tr}(\mathrm{Cov})$")
    ax.set_title("Figure 2: stationary bias vs step size")
    ax.legend()
    save_figure(fig, "fig2_bias_vs_stepsize")

    # Figure 3: ESS per gradient evaluation vs step size
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in ("ULA", "MALA", "KLMC"):
        ax.plot(step_sizes, results[name]["ess_per_grad"], label=name, **SAMPLER_STYLE[name])
    ax.set_xscale("log")
    ax.set_xlabel("step size h")
    ax.set_ylabel("min-coordinate ESS / gradient evaluation")
    ax.set_title("Figure 3: mixing efficiency vs step size")
    ax.legend()
    save_figure(fig, "fig3_ess_vs_stepsize")

    # Figure 4: MALA acceptance rate vs step size
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(step_sizes, results["MALA"]["accept_rate"], **SAMPLER_STYLE["MALA"])
    ax.axhline(0.574, color="gray", linestyle="--", linewidth=1,
               label="0.574 (optimal scaling heuristic)")
    ax.set_xscale("log")
    ax.set_xlabel("step size h")
    ax.set_ylabel("MALA acceptance rate")
    ax.set_title("Figure 4: MALA acceptance rate vs step size")
    ax.legend()
    save_figure(fig, "fig4_mala_acceptance_vs_stepsize")


if __name__ == "__main__":
    step_sizes, results = run_sweep()
    save_results("step_size_sweep", results)
    make_plots(step_sizes, results)
