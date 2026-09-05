"""Precision (condition number) scaling: reproduces Figure 7.

Non-asymptotic guarantees for ULA, MALA and KLMC are all stated in terms
of the condition number kappa = L / m of the log-density's Hessian
(L = smoothness / largest eigenvalue of the precision matrix, m =
strong-convexity / smallest eigenvalue): roughly, the number of gradient
evaluations needed scales like kappa for ULA/MALA and like sqrt(kappa)
for KLMC. This script fixes the dimension and sweeps kappa (by
constructing precision matrices whose eigenvalues span a wider and
wider range via ``make_ill_conditioned_gaussian``), using for each
sampler the step size theory prescribes as a function of L (which grows
with kappa here, since we hold m fixed... precisely, we hold the
*geometric mean* sqrt(L*m) = 1 fixed and let L = sqrt(kappa),
m = 1/sqrt(kappa), so L grows and m shrinks symmetrically as kappa
increases):

    h_ULA, h_MALA  = c / L
    h_KLMC         = c' / L,   gamma_KLMC = 2 sqrt(L)   (as elsewhere)

and measures the number of gradient evaluations required to reach a
fixed target ESS (equivalently, ESS per gradient evaluation, inverted)
as a function of kappa - the plot that is most directly comparable to
the theoretical kappa vs sqrt(kappa) complexity statements above.

Caveat: the kappa and sqrt(kappa) reference lines in the resulting
figure are theoretical slopes, plotted for comparison, not fits. With
the simple, kappa-independent friction/step-size heuristics used here
(gamma = 2 sqrt(L), h = 0.1 / sqrt(L)), KLMC's *slope* with kappa sits
between the sqrt(kappa) and kappa reference lines rather than cleanly
matching sqrt(kappa) - and, in this single-chain ESS-based measurement,
it is uniformly more expensive per effective sample than ULA/MALA
(likely reflecting both the extra velocity update per step and a
friction/step-size pair that is not actually kappa-optimal). Cheng et
al.'s and Dalalyan & Riou-Durand's sqrt(kappa) results are asymptotic,
non-asymptotic-worst-case guarantees under carefully kappa-tuned
hyperparameters and a Wasserstein-distance notion of mixing; reproducing
the crossover where KLMC overtakes ULA/MALA in wall-clock (gradient)
cost would need that same careful, kappa-dependent tuning rather than
the one-size-fits-all heuristics used elsewhere in this repo.

Run with:  python experiments/precision_scaling.py
"""
from __future__ import annotations

import numpy as np

from _common import SAMPLER_STYLE, plt, save_figure, save_results  # noqa: E402
from src.diagnostics.mixing import min_ess
from src.samplers.klmc import klmc_sample
from src.samplers.mala import mala_sample
from src.samplers.ula import ula_sample
from src.targets.gaussian import make_ill_conditioned_gaussian

SEED = 3
DIM = 20
CONDITION_NUMBERS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
N_STEPS = 20000
N_BURNIN = 4000
N_REPEATS = 3


def run_sweep():
    rng = np.random.default_rng(SEED)

    results = {"dim": DIM, "condition_numbers": CONDITION_NUMBERS}
    for name in ("ULA", "MALA", "KLMC"):
        results[name] = {"ess_per_grad": [], "grad_evals_per_ess": []}
    results["MALA"]["accept_rate"] = []

    for kappa in CONDITION_NUMBERS:
        target = make_ill_conditioned_gaussian(DIM, kappa, rng=rng)
        L = target.smoothness
        h = 0.5 / L
        # KLMC's sqrt(kappa) advantage over ULA/MALA's kappa scaling
        # comes precisely from being able to take steps of size
        # O(1/sqrt(L)) rather than O(1/L) while remaining a good
        # (numerically well-conditioned - see friction_sweep.py)
        # integrator, thanks to the momentum/friction pair damping the
        # stiff directions instead of the step size alone having to.
        h_klmc = 0.1 / np.sqrt(L)
        gamma_klmc = 2.0 * np.sqrt(L)
        x0 = np.zeros(DIM)

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
            m = float(np.mean(ess[name]))
            results[name]["ess_per_grad"].append(m)
            results[name]["grad_evals_per_ess"].append(1.0 / m if m > 0 else float("inf"))
        results["MALA"]["accept_rate"].append(float(np.mean(accept_rates)))

        print(f"kappa={kappa:6.1f}  ULA ess/grad={results['ULA']['ess_per_grad'][-1]:.4g}  "
              f"MALA ess/grad={results['MALA']['ess_per_grad'][-1]:.4g} "
              f"(acc={results['MALA']['accept_rate'][-1]:.2f})  "
              f"KLMC ess/grad={results['KLMC']['ess_per_grad'][-1]:.4g}")

    return results


def make_plots(results):
    kappas = np.array(results["condition_numbers"])

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in ("ULA", "MALA", "KLMC"):
        ax.plot(kappas, results[name]["grad_evals_per_ess"], label=name, **SAMPLER_STYLE[name])

    # Reference slopes: theory predicts ~kappa for ULA/MALA, ~sqrt(kappa) for KLMC.
    ref = results["ULA"]["grad_evals_per_ess"][0] / kappas[0]
    ax.plot(kappas, ref * kappas, "k--", linewidth=1, label=r"$\propto \kappa$")
    ref_sqrt = results["KLMC"]["grad_evals_per_ess"][0] / np.sqrt(kappas[0])
    ax.plot(kappas, ref_sqrt * np.sqrt(kappas), "k:", linewidth=1, label=r"$\propto \sqrt{\kappa}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"condition number $\kappa = L/m$")
    ax.set_ylabel("gradient evaluations per effective sample")
    ax.set_title("Figure 7: mixing cost vs condition number")
    ax.legend()
    save_figure(fig, "fig7_precision_scaling")


if __name__ == "__main__":
    results = run_sweep()
    save_results("precision_scaling", results)
    make_plots(results)
