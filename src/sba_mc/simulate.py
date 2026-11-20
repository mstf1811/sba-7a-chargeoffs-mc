"""Monte Carlo simulation functions (vectorised, R x n arrays)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mc_mle_poisson(lam: float, expo_ref: np.ndarray, n: int, R: int,
                   rng: np.random.Generator, block: int = 500):
    """Monte Carlo #1: MLE of lambda over R simulated datasets of size n.

    Reference world: Y_i ~ Poisson(lam * e_i), e_i drawn with replacement from expo_ref.
    Returns (lam_hat, crb) where crb = lam / sum(e) is the Cramer-Rao bound of each replicate.
    """
    lam_hat, crb = np.empty(R), np.empty(R)
    for d in range(0, R, block):
        r = min(block, R - d)
        e = rng.choice(expo_ref, size=(r, n), replace=True)
        Y = rng.poisson(lam * e)
        se = e.sum(axis=1)
        lam_hat[d:d + r] = Y.sum(axis=1) / se
        crb[d:d + r] = lam / se
    return lam_hat, crb


def summarize_mc(estimates: np.ndarray, true_value: float) -> pd.Series:
    """Bias, variance, MSE with Monte Carlo standard errors."""
    est = np.asarray(estimates, dtype=float)
    R = len(est)
    bias = est.mean() - true_value
    var = est.var(ddof=1)
    sq = (est - true_value) ** 2
    return pd.Series({
        "R": R,
        "bias": bias, "mc_se_bias": np.sqrt(var / R),
        "variance": var, "mc_se_variance": var * np.sqrt(2.0 / (R - 1)),
        "MSE": sq.mean(), "mc_se_MSE": sq.std(ddof=1) / np.sqrt(R),
    })


def mc_se_proportion(p_hat: float, R: int) -> float:
    """Monte Carlo standard error of an estimated proportion (coverage, size, power)."""
    return float(np.sqrt(p_hat * (1.0 - p_hat) / R))
