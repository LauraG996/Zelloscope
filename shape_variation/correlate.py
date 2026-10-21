"""Link shape (mode scores, dimensions) to cell structure and process data. Small data: rank correlations and a
shallow random forest with leave-one-out validation; everything is exploratory."""
import numpy as np
import pandas as pd
from scipy import stats


def spearman_table(features: pd.DataFrame, targets: pd.DataFrame, min_n: int = 6) -> pd.DataFrame:
    """Spearman rho and p-value for every feature x target pair with enough complete rows, |rho| descending."""
    rows = []
    for f in features.columns:
        for t in targets.columns:
            ok = features[f].notna() & targets[t].notna()
            x, y = features.loc[ok, f].astype(float), targets.loc[ok, t].astype(float)
            if ok.sum() < min_n or x.nunique() < 2 or y.nunique() < 2:
                continue
            rho, p = stats.spearmanr(x, y)
            rows.append(dict(feature=f, target=t, rho=round(float(rho), 3), p=round(float(p), 4), n=int(ok.sum())))
    df = pd.DataFrame(rows, columns=["feature", "target", "rho", "p", "n"])
    if len(df):
        # Benjamini-Hochberg so that "significant" accounts for testing many pairs
        p = df["p"].to_numpy()
        order = np.argsort(p)
        q_sorted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        q = np.empty(len(p))
        q[order] = np.minimum(q_sorted, 1.0)
        df["q"] = q.round(4)
        df = df.reindex(df["rho"].abs().sort_values(ascending=False).index)
    return df


def forest_importance(features: pd.DataFrame, target: pd.Series, seed: int = 0) -> dict | None:
    """Shallow random forest: leave-one-out R^2 (honest skill) and permutation importance on the full fit."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    ok = target.notna() & features.notna().all(axis=1)
    x, y = features.loc[ok].astype(float), target.loc[ok].astype(float)
    if len(y) < 8 or x.shape[1] < 1 or y.nunique() < 3:
        return None
    rf = RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=2, random_state=seed)
    pred = cross_val_predict(rf, x, y, cv=LeaveOneOut())
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - ((y - pred) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")
    rf.fit(x, y)
    imp = permutation_importance(rf, x, y, n_repeats=20, random_state=seed)
    order = np.argsort(imp.importances_mean)[::-1]
    return dict(n=int(len(y)), loo_r2=round(r2, 3),
                importance=[(x.columns[i], round(float(imp.importances_mean[i]), 4)) for i in order])
