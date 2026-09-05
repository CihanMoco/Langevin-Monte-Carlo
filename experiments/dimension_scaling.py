"""Dimension scaling: reproduces Figure 6.

We fix the target's condition number (so the smoothness/strong-convexity
constants L, m of the log-density Hessian do not change with dimension -
``make_ill_conditioned_gaussian`` log-spaces the precision matrix's
eigenvalues between the same 1/sqrt(kappa) and sqrt(kappa) regardless of
d) and sweep the dimension d, holding each sampler's step size *fixed*
(i.e. not re-tuned per dimension).

Two effects are then visible:

    Figure 6a - min-coordinate ESS per gradient evaluation vs d, for
    ULA/MALA/KLMC. ULA and KLMC have no accept/reject step, so with L, m
    unchanged their per-coordinate mixing rate does not intrinsically
    degrade with d.

    Figure 6b - MALA's acceptance rate vs d, at that same fixed step
    size. MALA's Metropolis correction compares the target density at a
    proposal that accumulates independent noise in every one of the d
    coordinates, so - unlike ULA/KLMC - its acceptance probability
    degrades as d grows unless h is shrunk (the classical optimal-scaling
    result of Roberts & Rosenthal (1998) prescribes h = O(d^{-1/3}) for
    MALA). This is exactly the mechanism that makes MALA's per-gradient
    mixing efficiency fall off with dimension in the left panel even
    though ULA/KLMC's does not.

Run with:  python experiments/dimension_scaling.py
"""
from __future__ import annotations

import numpy as np

from _common import SAMPLER_STYLE, plt, save_figure, save_results  # noqa: E402
from src.diagnostics.mixing import min_ess
from src.samplers.klmc import klmc_sample
from src.samplers.mala import mala_sample
from src.samplers.ula import ula_sample
from src.targets.gaussian import make_ill_conditioned_gaussian

SEED = 2
CONDITION_NUMBER = 10.0
DIMS = [2, 5, 10, 20, 50, 100, 200]
N_STEPS = 20000
N_BURNIN = 4000
N_REPEATS = 3


def run_sweep():
    rng = np.random.default_rng(SEED)

    results = {"dims": DIMS, "condition_number": CONDITION_NUMBER}
    for name in ("ULA", "MALA", "KLMC"):
        results[name] = {"ess_per_grad": []}
    results["MALA"]["accept_rate"] = []

    for d in DIMS:
        target = make_ill_conditioned_gaussian(d, CONDITION_NUMBER, rng=rng)
        L = target.smoothness
        # Fixed (not re-tuned per dimension) step sizes: L is the same
        # for every d by construction, so this isolates the effect of d
        # itself rather than re-deriving a per-dimension step size.
        h = 0.5 / L
        gamma_klmc = 2.0 * np.sqrt(L)
        h_klmc = 0.3 / L
        x0 = np.zeros(d)

        ess = {"ULA": [], "MALA": [], "KLMC": []}
        accept_rates = []

        for _ in range(N_REPEATS):
            seed_rng = np.random.default_rng(rng.integers(1 << 32))

            chain = ula_sample(target.grad_log_prob, x0, h, N_STEPS, rng=seed_rng)
            ess["ULA"].append(min_ess(chain[N_BURNIN:]) / (N_STEPS - N_BURNIN))

            mres = mala_sample(target.log_prob, target.grad_log_prob, x0, h, N_STEPS, rng=seed_rng)
            ess["MALA"].append(min_ess(mres.chain[N_BURNIN:]) / (N_STEPS - N_BURNIN))
            accept_rates.append(mres.accept_rate)

            kres = klmc_sample(target.grad_log_prob, x0, h_klmc, N_STEPS, gamma=gamma_klmc, rng=seed_rng)
            ess["KLMC"].append(min_ess(kres.x_chain[N_BURNIN:]) / (N_STEPS - N_BURNIN))

        for name in ("ULA", "MALA", "KLMC"):
            results[name]["ess_per_grad"].append(float(np.mean(ess[name])))
        results["MALA"]["accept_rate"].append(float(np.mean(accept_rates)))

        print(f"d={d:4d}  ULA ess/grad={results['ULA']['ess_per_grad'][-1]:.4g}  "
              f"MALA ess/grad={results['MALA']['ess_per_grad'][-1]:.4g} "
              f"(acc={results['MALA']['accept_rate'][-1]:.2f})  "
              f"KLMC ess/grad={results['KLMC']['ess_per_grad'][-1]:.4g}")

    return results


def make_plots(results):
    dims = results["dims"]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in ("ULA", "MALA", "KLMC"):
        ax.plot(dims, results[name]["ess_per_grad"], label=name, **SAMPLER_STYLE[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("min-coordinate ESS / gradient evaluation")
    ax.set_title("Figure 6a: mixing efficiency vs dimension\n(fixed condition number, fixed step size)")
    ax.legend()
    save_figure(fig, "fig6a_ess_vs_dimension")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(dims, results["MALA"]["accept_rate"], **SAMPLER_STYLE["MALA"])
    ax.set_xscale("log")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("MALA acceptance rate")
    ax.set_title("Figure 6b: MALA acceptance rate vs dimension\n(step size not re-tuned per dimension)")
    save_figure(fig, "fig6b_mala_acceptance_vs_dimension")


if __name__ == "__main__":
    results = run_sweep()
    save_results("dimension_scaling", results)
    make_plots(results)
