"""Split / Mondrian conformal calibration for predictive intervals.

Reference: Vovk et al., *Algorithmic Learning in a Random World* (split conformal);
Romano et al. 2019 (CQR); Mondrian (class/regime-conditional) conformal: Vovk 2012.

Model-agnostic and pure-numpy.  Given a held-out **calibration** set (disjoint from
both proper-train and test) of point predictions ``mu`` with a per-sample scale
``sigma`` and the calibration targets, it computes the conformal multiplier ``q`` and
emits intervals ``[mu - q*sigma, mu, mu + q*sigma]`` with finite-sample marginal
coverage ≥ 1-alpha (split conformal).  The Mondrian variant computes ``q`` per
observable bin so coverage holds *within* each bin — the lever for the left-tail
under-coverage seen under k-fold (TabM/XGB ≈ 0.66 vs nominal 0.90).

Bins MUST be defined on observable quantities (features or predictions), never on the
target ``y`` — a y-defined bin (e.g. "bottom 5% of y") is unavailable at test time.
``make_quantile_bins`` on predicted ``mu`` is the default proxy for the low-prediction
tail; cloud-proximity / AOD bins also work if passed in.

Output is always [N, 3] = (q05, q50=mu, q95), monotone by construction (crossing = 0).
"""

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float = 0.10) -> float:
    """Finite-sample conformal quantile of nonconformity ``scores``.

    Uses the (1-alpha)(n+1)/n empirical level (the standard split-conformal
    correction).  Returns +inf if there are no scores (interval becomes infinite,
    i.e. no calibration → no guarantee).
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        return float('inf')
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method='higher'))


def normalized_scores(y, mu, sigma, eps: float = 1e-6) -> np.ndarray:
    """Normalized absolute residual |y - mu| / max(sigma, eps)."""
    y, mu, sigma = map(lambda a: np.asarray(a, dtype=float), (y, mu, sigma))
    return np.abs(y - mu) / np.maximum(sigma, eps)


def _intervals(mu, sigma, q) -> np.ndarray:
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    lo = mu - q * sigma
    hi = mu + q * sigma
    return np.column_stack([lo, mu, hi])


def split_conformal(cal_y, cal_mu, cal_sigma, test_mu, test_sigma, alpha: float = 0.10):
    """Global split conformal.  Returns (preds [N,3], q)."""
    q = conformal_quantile(normalized_scores(cal_y, cal_mu, cal_sigma), alpha)
    return _intervals(test_mu, test_sigma, q), q


def make_quantile_bins(values, n_bins: int, edges=None):
    """Assign each value to one of ``n_bins`` quantile bins.

    If ``edges`` is None they are computed from ``values`` (use the calibration
    distribution, then pass the returned edges to bin the test set identically).
    Returns (bin_idx [N], edges [n_bins+1]).
    """
    values = np.asarray(values, dtype=float)
    if edges is None:
        edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.asarray(edges, dtype=float)
        edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    return idx.astype(int), edges


def regime_alphas(cal_bin, is_near, near_alpha: float, far_alpha: float,
                  frac: float = 0.5) -> dict:
    """Build a per-bin alpha map: a bin gets ``near_alpha`` if at least ``frac`` of
    its calibration points are flagged ``is_near`` (e.g. cloud distance <= 10 km),
    else ``far_alpha``.

    Used to *deliberately over-cover* the near-cloud regime (near_alpha < far_alpha
    ⇒ higher target there): the only way to lift coverage on the outcome-defined
    tail, which conformal cannot guarantee at a flat target by construction.
    """
    cal_bin = np.asarray(cal_bin, dtype=int)
    is_near = np.asarray(is_near, dtype=bool)
    return {int(b): (near_alpha if is_near[cal_bin == b].mean() >= frac else far_alpha)
            for b in np.unique(cal_bin)}


def mondrian_conformal(cal_y, cal_mu, cal_sigma, cal_bin,
                       test_mu, test_sigma, test_bin,
                       alpha: float = 0.10, min_per_bin: int = 50,
                       bin_alpha: 'dict | None' = None):
    """Regime-conditional conformal: a separate ``q`` per bin.

    Bins with fewer than ``min_per_bin`` calibration points fall back to the global
    ``q`` (avoids degenerate intervals from tiny bins).  Returns (preds [N,3], q_by_bin).

    ``bin_alpha`` optionally overrides the scalar ``alpha`` per bin (see
    ``regime_alphas``); bins absent from the dict use the scalar ``alpha``.  Each
    bin's global fallback is also computed at that bin's alpha.
    """
    cal_bin = np.asarray(cal_bin, dtype=int)
    test_bin = np.asarray(test_bin, dtype=int)
    scores = normalized_scores(cal_y, cal_mu, cal_sigma)

    def _alpha(b):
        return bin_alpha.get(int(b), alpha) if bin_alpha else alpha

    test_mu = np.asarray(test_mu, dtype=float)
    test_sigma = np.asarray(test_sigma, dtype=float)
    lo = np.empty_like(test_mu)
    hi = np.empty_like(test_mu)
    q_by_bin = {}
    for b in np.unique(test_bin):
        a = _alpha(b)
        m = cal_bin == b
        q = conformal_quantile(scores[m], a) if m.sum() >= min_per_bin \
            else conformal_quantile(scores, a)
        q_by_bin[int(b)] = float(q)
        tm = test_bin == b
        lo[tm] = test_mu[tm] - q * test_sigma[tm]
        hi[tm] = test_mu[tm] + q * test_sigma[tm]
    return np.column_stack([lo, test_mu, hi]), q_by_bin
