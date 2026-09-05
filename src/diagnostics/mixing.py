"""Mixing diagnostics: FFT-based autocorrelation and effective sample size.

Computing the empirical autocorrelation function (ACF) of a chain of
length n by direct summation costs O(n^2). Instead we use the standard
trick of computing it via the FFT in O(n log n): the ACF is the inverse
Fourier transform of the power spectral density, so zero-padding the
(mean-centred) chain to avoid circular-correlation artefacts and taking
|FFT|^2 followed by an inverse FFT recovers the full autocovariance
sequence in one shot.

The effective sample size (ESS) is then estimated from the ACF using
Geyer's initial positive sequence estimator (Geyer, 1992, "Practical
Markov Chain Monte Carlo"), which is the standard, robust way to turn a
noisy empirical ACF into a single integrated autocorrelation time
without having to hand-pick a truncation lag.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def autocorrelation_fft(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Empirical autocorrelation function of a 1D chain, via FFT.

    Parameters
    ----------
    x : ndarray, shape (n,)
        A single scalar chain (e.g. one coordinate, or a scalar summary
        such as the log-density).
    max_lag : int, optional
        Number of lags to return (including lag 0). Defaults to n.

    Returns
    -------
    acf : ndarray, shape (max_lag,)
        acf[0] == 1.0 by construction; acf[k] is the lag-k autocorrelation.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if max_lag is None:
        max_lag = n
    max_lag = min(max_lag, n)

    x = x - x.mean()
    if np.allclose(x, 0.0):
        # Degenerate (constant) chain: define acf = 1 at lag 0, 0 elsewhere.
        acf = np.zeros(max_lag)
        acf[0] = 1.0
        return acf

    # Zero-pad to at least 2n (and to a power of two for FFT speed) so the
    # implicit circular convolution of the FFT does not wrap around and
    # contaminate the autocovariance estimate.
    size = 2 * n
    nfft = 1
    while nfft < size:
        nfft *= 2

    f = np.fft.rfft(x, n=nfft)
    power = f * np.conjugate(f)
    acov = np.fft.irfft(power, n=nfft)[:n].real

    # Bias correction for the fact that each lag k average is over n - k
    # (not n) pairs.
    lags = np.arange(n)
    acov /= (n - lags)

    acf = acov / acov[0]
    return acf[:max_lag]


def integrated_autocorr_time(x: np.ndarray) -> float:
    """Integrated autocorrelation time tau = 1 + 2 * sum_{k=1}^K rho_k,
    with the summation cutoff K chosen via Geyer's initial positive
    sequence rule: pair consecutive lags (rho_{2m-1} + rho_{2m}) and sum
    pairs only while their running sum stays positive (and, for the
    stricter "initial monotone" variant, non-increasing).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n < 4:
        return 1.0

    acf = autocorrelation_fft(x)

    # Pair up (1,2), (3,4), ... and accumulate while each pair sum is
    # positive and the sequence of pair sums is non-increasing (Geyer's
    # initial monotone sequence estimator).
    n_pairs = (len(acf) - 1) // 2
    running_sum = 0.0
    prev_pair_sum = np.inf
    for m in range(n_pairs):
        k = 1 + 2 * m
        pair_sum = acf[k] + acf[k + 1]
        pair_sum = min(pair_sum, prev_pair_sum)  # enforce monotonicity
        if pair_sum <= 0:
            break
        running_sum += pair_sum
        prev_pair_sum = pair_sum

    tau = 1.0 + 2.0 * running_sum
    return max(tau, 1.0)


def effective_sample_size(x: np.ndarray) -> float:
    """Effective sample size of a scalar chain, ESS = n / tau."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    tau = integrated_autocorr_time(x)
    return n / tau


def effective_sample_size_per_dim(chain: np.ndarray) -> np.ndarray:
    """Apply :func:`effective_sample_size` to every column of a
    (n_samples, dim) chain."""
    chain = np.asarray(chain, dtype=float)
    return np.array([effective_sample_size(chain[:, j]) for j in range(chain.shape[1])])


def min_ess(chain: np.ndarray) -> float:
    """Conservative summary: the smallest per-coordinate ESS."""
    return float(np.min(effective_sample_size_per_dim(chain)))
