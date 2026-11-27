"""Novelty 1: power and sample-size rule for the reference-calibrated Pillai gate.

Model. For p- and d-dimensional payloads with population squared canonical correlations
rho_k^2 (theta = sum rho_k^2), mispairing a fraction q of a batch by derangement scales the
cross-covariance by (1 - q), so rho_k^2 -> (1 - q)^2 rho_k^2. The batch Pillai trace V of N
records is approximately normal with

    mean  m(q) = theta_q + p d / N,                     theta_q = (1 - q)^2 theta
    var   v(q) = 4 sum_k rho_kq^2 (1 - rho_kq^2)^2 / N + (2 p d / N^2)(1 - (p + d + 1)/N)^2

(first term: delta method for the sample canonical correlations; second: the null part).
With the conformal threshold t = m(0) - z_{1-alpha} sqrt(v(0)), the predicted power is

    power(N, q) = Phi( (t - m(q)) / sqrt(v(q)) ).

Sample-size rule: the smallest N with power >= 1 - beta (solved numerically). For weak
dependence (rho_k^2 << 1, N theta >> p d) it reduces to

    N* ~= 4 (z_{1-alpha} + (1 - q) z_{1-beta})^2 / (theta q^2 (2 - q)^2).

Two-population form (reference and test data differ, e.g. different subjects): use theta_ref
for m(0), v(0) and theta_test for the test batch; a finite pool of n_pool records sampled
without replacement multiplies the variance by (1 - N / n_pool).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .gate import _orth


def spectrum(x, y):
    """Sample squared canonical correlations."""
    return np.linalg.svd(_orth(x).T @ _orth(y), compute_uv=False) ** 2


def moments(rho2, N, q, p=4, d=8, pool=None):
    rq = np.asarray(rho2, float) * (1 - q) ** 2
    m = rq.sum() + p * d / N
    v = 4 * np.sum(rq * (1 - rq) ** 2) / N + (2 * p * d / N ** 2) * (1 - (p + d + 1) / N) ** 2
    if pool:
        v *= max(1 - N / pool, 1e-6)
    return m, v


def power(rho2_test, N, q, alpha=0.05, p=4, d=8, rho2_ref=None, pool_ref=None, pool_test=None):
    ref = rho2_test if rho2_ref is None else rho2_ref
    m0, v0 = moments(ref, N, 0.0, p, d, pool_ref)
    mq, vq = moments(rho2_test, N, q, p, d, pool_test)
    t = m0 - norm.ppf(1 - alpha) * np.sqrt(v0)
    return float(norm.cdf((t - mq) / np.sqrt(vq)))


def clean_false_block(rho2_test, N, alpha=0.05, p=4, d=8, rho2_ref=None, pool_ref=None, pool_test=None):
    return power(rho2_test, N, 0.0, alpha, p, d, rho2_ref, pool_ref, pool_test)


def sample_size(rho2, q, alpha=0.05, beta=0.2, p=4, d=8, n_max=10 ** 6):
    """Smallest N (>= p + d + 2) with predicted power >= 1 - beta; None if not reached by n_max."""
    lo = p + d + 2
    if power(rho2, n_max, q, alpha, p, d) < 1 - beta:
        return None
    hi = n_max
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if power(rho2, mid, q, alpha, p, d) >= 1 - beta:
            hi = mid
        else:
            lo = mid
    return hi


def sample_size_weak(theta, q, alpha=0.05, beta=0.2):
    za, zb = norm.ppf(1 - alpha), norm.ppf(1 - beta)
    return 4 * (za + (1 - q) * zb) ** 2 / (theta * q ** 2 * (2 - q) ** 2)
