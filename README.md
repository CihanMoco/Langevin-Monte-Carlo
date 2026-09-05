# Langevin Monte Carlo: From Theory to Implementation and Kinetic Extensions

Independent Python implementation of the three Langevin-based MCMC samplers 
studied in our group research project at Imperial College London 
(supervised by Randolf Altmeyer).

The original group project produced a joint paper (see `paper/`). 
This repository contains my own reimplementation of all samplers and 
experiments from scratch.

## Samplers
- **ULA** — Unadjusted Langevin Algorithm (Euler-Maruyama discretisation)
- **MALA** — Metropolis-Adjusted Langevin Algorithm
- **KLMC** — Kinetic Langevin Monte Carlo (Dalalyan & Riou-Durand, 2020)

## Repository Structure
```
langevin-monte-carlo/
├── paper/                       # the group paper (add langevin_monte_carlo.pdf here)
├── src/
│   ├── samplers/                # ula.py, mala.py, klmc.py
│   ├── targets/                 # gaussian.py, linear_regression.py, logistic_regression.py
│   └── diagnostics/             # mixing.py (FFT autocorrelation + ESS)
├── experiments/                 # step_size_sweep.py, friction_sweep.py,
│                                 # dimension_scaling.py, precision_scaling.py
├── plots/                       # generated figures (PNG)
└── results/                     # generated numerical output (JSON)
```

## Key Results
- `plots/fig2_bias_vs_stepsize.png` — ULA's stationary bias grows with step size while MALA stays unbiased.
- `plots/fig4_mala_acceptance_vs_stepsize.png` — the classic MALA acceptance-rate / step-size trade-off.
- `plots/fig6b_mala_acceptance_vs_dimension.png` — MALA's acceptance rate degrading with dimension at a fixed (non-rescaled) step size.
- `plots/fig7_precision_scaling.png` — mixing cost vs the target's condition number, against the theoretical kappa / sqrt(kappa) reference slopes.

## How to Run
```
pip install -r requirements.txt
python experiments/step_size_sweep.py     # Figures 2-4
python experiments/friction_sweep.py      # Figure 5
python experiments/dimension_scaling.py   # Figure 6
python experiments/precision_scaling.py   # Figure 7
```
Each script writes its figures to `plots/` and its numerical results to `results/`.

## References
- Dalalyan & Riou-Durand (2020), "On sampling from a log-concave density 
  using kinetic Langevin diffusions"
