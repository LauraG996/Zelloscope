"""Shape change after extrusion: exponential relaxation fits for parts imaged repeatedly.

Model per part p:   y_p(t) = c_p + A * exp(-t / tau)
  c_p    - the part's final (stabilised) value
  A, tau - amplitude and time constant shared by all parts (pooled: robust with few images per part)
"""
import numpy as np
from scipy.optimize import least_squares


def fit_pooled(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """series: part -> (t (min since extrusion), y). Needs >= 3 points in total more than the parameters."""
    parts = sorted(series)
    t_all = [np.asarray(series[p][0], float) for p in parts]
    y_all = [np.asarray(series[p][1], float) for p in parts]
    n_par = len(parts) + 2
    n_pts = sum(len(t) for t in t_all)
    if n_pts < n_par + 2:
        raise ValueError(f"too few points ({n_pts}) for {n_par} parameters")
    t_span = max(np.ptp(np.concatenate(t_all)), 1e-6)

    def resid(theta):
        a, log_tau = theta[:2]
        tau = np.exp(log_tau)
        return np.concatenate([y - (theta[2 + i] + a * np.exp(-t / tau)) for i, (t, y) in enumerate(zip(t_all, y_all))])

    a0 = np.mean([y[0] - y[-1] for y in y_all])
    x0 = np.concatenate([[a0 if a0 else 1.0, np.log(t_span / 3)], [y[-1] for y in y_all]])
    sol = least_squares(resid, x0)
    a, tau = sol.x[0], float(np.exp(sol.x[1]))
    r = sol.fun
    y_cat = np.concatenate(y_all)
    ss_tot = ((y_cat - y_cat.mean()) ** 2).sum()
    # parameter standard errors from the Jacobian
    dof = max(1, n_pts - n_par)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * (r @ r) / dof
        tau_se = float(tau * np.sqrt(cov[1, 1]))          # delta method on log(tau)
    except np.linalg.LinAlgError:
        tau_se = float("nan")
    return dict(amplitude=float(a), tau=tau, tau_se=tau_se, final={p: float(c) for p, c in zip(parts, sol.x[2:])},
                r2=float(1 - (r @ r) / ss_tot) if ss_tot > 0 else float("nan"), n_points=n_pts, n_parts=len(parts))


def settle_time(tau: float, fraction: float) -> float:
    """Time after which `fraction` of the total change has happened."""
    return float(-tau * np.log(1 - fraction))


def predict_final(t_early: float, y_early: float, amplitude: float, tau: float) -> float:
    return float(y_early - amplitude * np.exp(-t_early / tau))


def early_prediction_check(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Leave-one-part-out: fit A, tau on the other parts, predict this part's last value from its first image."""
    errors = {}
    for p in series:
        others = {q: v for q, v in series.items() if q != p}
        try:
            fit = fit_pooled(others)
        except ValueError:
            continue
        t, y = series[p]
        order = np.argsort(t)
        pred = predict_final(t[order][0], y[order][0], fit["amplitude"], fit["tau"])
        errors[p] = float(pred - y[order][-1])
    e = np.array(list(errors.values()))
    no_model = {p: float(series[p][1][np.argsort(series[p][0])][0] - series[p][1][np.argsort(series[p][0])][-1])
                for p in errors}
    return dict(errors=errors, rmse=float(np.sqrt((e ** 2).mean())) if len(e) else float("nan"),
                rmse_without_model=float(np.sqrt(np.mean(np.square(list(no_model.values()))))) if no_model else float("nan"))
