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
├── paper/                       # research paper
├── src/
│   ├── samplers/                # ula.py, mala.py, klmc.py
│   ├── targets/                 # gaussian.py, linear_regression.py, logistic_regression.py
│   └── diagnostics/             # mixing.py (FFT autocorrelation + ESS)
├── experiments/                 # step_size_sweep.py, friction_sweep.py,
│                                # dimension_scaling.py, precision_scaling.py
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
1. Arnak S. Dalalyan and Lionel Riou-Durand. On sampling from a log-concave density using kinetic Langevin diffusions. *Bernoulli*, 26(3):1956–1988, 2020. doi: 10.3150/19-BEJ1178.
2. Gareth O. Roberts and Richard L. Tweedie. Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*, 2(4):341–363, 1996.
3. Arnak S. Dalalyan. Theoretical guarantees for approximate sampling from smooth and log-concave densities. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 79(3):651–676, 2017. doi: 10.1111/rssb.12183.
4. Alain Durmus and Éric Moulines. Nonasymptotic convergence analysis for the unadjusted Langevin algorithm. *The Annals of Applied Probability*, 27(3):1551–1587, 2017. doi: 10.1214/16-AAP1238.
5. sampling-using-SDEs. https://github.com/maxcyn/sampling-using-SDEs, 2026. GitHub repository.
6. Andrew Duncan. M4A44: Computational stochastic processes. Lecture notes, Imperial College London, 2016.
7. Xiang Cheng, Niladri S. Chatterji, Peter L. Bartlett, and Michael I. Jordan. Underdamped Langevin MCMC: A non-asymptotic analysis. In *Proceedings of the 31st Conference on Learning Theory*, volume 75 of *PMLR*, pages 300–323, 2018.
8. Edward Nelson. *Dynamical Theories of Brownian Motion*. Princeton University Press, 1967. doi: 10.1515/9780691219615.
