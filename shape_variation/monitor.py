"""Drift (SPC individuals charts) and anomaly scores in shape-model space."""
import numpy as np
from scipy import stats


def i_chart(values, n_sigma: float = 3.0, run_length: int = 8, trend_length: int = 6, baseline=None) -> dict:
    """Individuals chart: centre line and limits from the baseline points (default all), sigma from the average
    moving range (MR-bar / 1.128). Flags points outside the limits, runs on one side, and steady trends."""
    y = np.asarray(values, float)
    base = y if baseline is None else y[np.asarray(baseline)]
    base = base[np.isfinite(base)]
    cl = float(base.mean()) if len(base) else float("nan")
    mr = np.abs(np.diff(base))
    sigma = float(mr.mean() / 1.128) if len(mr) else float("nan")
    ucl, lcl = cl + n_sigma * sigma, cl - n_sigma * sigma
    violations = []
    for i, v in enumerate(y):
        if np.isfinite(v) and (v > ucl or v < lcl):
            violations.append(dict(index=i, rule=f"beyond {n_sigma:g} sigma"))
    side = np.sign(y - cl)
    for i in range(run_length - 1, len(y)):
        w = side[i - run_length + 1: i + 1]
        if np.all(w == w[0]) and w[0] != 0:
            violations.append(dict(index=i, rule=f"{run_length} in a row on one side"))
    d = np.sign(np.diff(y))
    for i in range(trend_length - 1, len(y)):
        w = d[i - trend_length + 1: i]
        if len(w) == trend_length - 1 and np.all(w == w[0]) and w[0] != 0:
            violations.append(dict(index=i, rule=f"{trend_length} rising/falling"))
    return dict(cl=cl, ucl=ucl, lcl=lcl, sigma=sigma, violations=violations)


def t2_limit(n_ref: int, k: int, alpha: float) -> float:
    """Hotelling T^2 upper limit for a new observation, mean and covariance estimated from n_ref parts."""
    if n_ref - k < 1:
        return float("nan")
    f = stats.f.ppf(1 - alpha, k, n_ref - k)
    return float(k * (n_ref + 1) * (n_ref - 1) / (n_ref * (n_ref - k)) * f)


def t2_leave_one_out(scores: np.ndarray, k: int, alpha: float, reference=None) -> dict:
    """T^2 (squared Mahalanobis distance) of every part in the first k mode scores, each against the mean and
    covariance of the *other* reference parts, so an abnormal part cannot hide itself. k is reduced when there
    are too few parts for a covariance estimate. Also returns each mode's share of the T^2 (which mode makes a
    part abnormal)."""
    n = len(scores)
    ref = np.ones(n, bool) if reference is None else np.asarray(reference, bool)
    # A covariance of k modes needs several parts per mode to be stable: at most one mode per 3 reference parts.
    k = int(max(1, min(k, (ref.sum() - 1) // 3)))
    x = scores[:, :k]
    t2, limit, contrib = np.zeros(n), np.zeros(n), np.zeros((n, k))
    for i in range(n):
        others = ref.copy()
        others[i] = False
        xr = x[others]
        mu = xr.mean(axis=0)
        cov = np.atleast_2d(np.cov(xr, rowvar=False))
        inv = np.linalg.pinv(cov)
        d = x[i] - mu
        t2[i] = float(d @ inv @ d)
        # with (nearly) uncorrelated PCA scores, each mode's share is ~ d_j^2 / var_j
        contrib[i] = d * (inv @ d)
        limit[i] = t2_limit(int(others.sum()), k, alpha)
    return dict(k=k, t2=t2, limit=limit, abnormal=t2 > limit, contributions=contrib)


def spe_limit(spe_ref: np.ndarray, alpha: float) -> float:
    """Box's approximation for the limit of a squared prediction error: g * chi2(h), matched mean/variance."""
    m, v = float(np.mean(spe_ref)), float(np.var(spe_ref, ddof=1)) if len(spe_ref) > 1 else 0.0
    if m <= 0 or v <= 0:
        return float("nan")
    g, h = v / (2 * m), 2 * m * m / v
    return float(g * stats.chi2.ppf(1 - alpha, h))


def spe_leave_one_out(points: np.ndarray, k: int) -> np.ndarray:
    """Squared residual (mm^2, summed over all points) of each part reconstructed from the first k modes of a PCA
    fitted to the *other* parts: shape change the model has not seen."""
    n = len(points)
    x = points.reshape(n, -1)
    out = np.zeros(n)
    for i in range(n):
        others = np.delete(x, i, axis=0)
        mu = others.mean(axis=0)
        _, _, vt = np.linalg.svd(others - mu, full_matrices=False)
        v = vt[: min(k, len(vt))]
        d = x[i] - mu
        out[i] = float(((d - (d @ v.T) @ v) ** 2).sum())
    return out
