from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _welch_df(g1: pd.Series, g2: pd.Series) -> float:
    v1 = g1.var(ddof=1) / len(g1)
    v2 = g2.var(ddof=1) / len(g2)
    denom = (v1**2 / (len(g1) - 1)) + (v2**2 / (len(g2) - 1))
    return float((v1 + v2) ** 2 / denom) if denom else np.nan


def _ttest_ci(g1: pd.Series, g2: pd.Series, *, equal_var: bool, confidence_level: float = 0.95) -> tuple[float, float]:
    diff = float(g1.mean() - g2.mean())
    if equal_var:
        df = len(g1) + len(g2) - 2
        pooled = ((len(g1) - 1) * g1.var(ddof=1) + (len(g2) - 1) * g2.var(ddof=1)) / df
        se = float(np.sqrt(pooled * (1 / len(g1) + 1 / len(g2))))
    else:
        df = _welch_df(g1, g2)
        se = float(np.sqrt(g1.var(ddof=1) / len(g1) + g2.var(ddof=1) / len(g2)))
    crit = stats.t.ppf(1 - (1 - confidence_level) / 2, df)
    return diff - crit * se, diff + crit * se


def t_test(
    group1: pd.Series,
    group2: pd.Series,
    *,
    equal_var: bool = False,
) -> pd.DataFrame:
    """Welch two-sample t-test, analogous to R's t.test(var.equal=FALSE)."""
    g1 = pd.to_numeric(group1, errors="coerce").dropna()
    g2 = pd.to_numeric(group2, errors="coerce").dropna()
    if len(g1) < 2 or len(g2) < 2:
        raise ValueError("t_test() requires at least two non-missing observations in each group.")

    res = stats.ttest_ind(g1, g2, equal_var=equal_var)
    df = float(getattr(res, "df", len(g1) + len(g2) - 2 if equal_var else _welch_df(g1, g2)))
    if hasattr(res, "confidence_interval"):
        ci = res.confidence_interval(confidence_level=0.95)
        ci_low, ci_high = float(ci.low), float(ci.high)
    else:
        ci_low, ci_high = _ttest_ci(g1, g2, equal_var=equal_var)

    return pd.DataFrame(
        [
            {
                "t": round(float(res.statistic), 5),
                "df": round(df, 2),
                "p_value": float(res.pvalue),
                "mean_1": round(float(g1.mean()), 7),
                "mean_2": round(float(g2.mean()), 7),
                "difference": round(float(g1.mean() - g2.mean()), 7),
                "ci_lower": round(ci_low, 7),
                "ci_upper": round(ci_high, 7),
            }
        ]
    )


def difference_in_means(
    formula: str,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Difference-in-means estimator, analogous to estimatr::difference_in_means()."""
    import patsy

    y, X = patsy.dmatrices(formula, data=data, return_type="dataframe")
    outcome_col = y.columns[0]
    treat_cols = [c for c in X.columns if "Intercept" not in c]
    if len(treat_cols) != 1:
        raise ValueError(
            "difference_in_means() expects exactly one binary treatment variable. "
            "For multiple covariates use lm_robust()."
        )

    treat_col = treat_cols[0]
    outcome = pd.to_numeric(y[outcome_col], errors="coerce")
    treat = pd.to_numeric(X[treat_col], errors="coerce")
    valid = outcome.notna() & treat.notna()
    outcome = outcome.loc[valid]
    treat = treat.loc[valid].astype(int)
    values = set(treat.unique())
    if not values.issubset({0, 1}) or len(values) != 2:
        raise ValueError("difference_in_means() expects a binary treatment coded 0/1.")

    g1 = outcome[treat == 1]
    g0 = outcome[treat == 0]
    res = stats.ttest_ind(g1, g0, equal_var=False)
    estimate = float(g1.mean() - g0.mean())
    if hasattr(res, "confidence_interval"):
        ci = res.confidence_interval(confidence_level=0.95)
        ci_low, ci_high = float(ci.low), float(ci.high)
    else:
        ci_low, ci_high = _ttest_ci(g1, g0, equal_var=False)
    t_stat = float(res.statistic)
    se = estimate / t_stat if t_stat else float(np.sqrt(g1.var(ddof=1) / len(g1) + g0.var(ddof=1) / len(g0)))
    df = float(getattr(res, "df", _welch_df(g1, g0)))

    return pd.DataFrame(
        [
            {
                "term": treat_col,
                "estimate": round(estimate, 7),
                "se": round(float(se), 7),
                "t": round(t_stat, 5),
                "df": round(df, 2),
                "p_value": float(res.pvalue),
                "ci_lower": round(ci_low, 7),
                "ci_upper": round(ci_high, 7),
                "n": int(len(outcome)),
            }
        ]
    )
