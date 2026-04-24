from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from .core import MatchResult, match_data


def _as_numeric_frame(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    x = df[covariates].copy()
    return pd.get_dummies(x, drop_first=False, dtype=float)


def _weighted_mean(x: pd.Series, w: pd.Series | None = None) -> float:
    if len(x) == 0:
        return np.nan
    if w is None:
        return float(x.mean())
    if len(w) == 0 or float(w.sum()) == 0:
        return np.nan
    return float(np.average(x, weights=w))


def _weighted_var(x: pd.Series, w: pd.Series | None = None) -> float:
    if len(x) == 0:
        return np.nan
    if w is None:
        return float(x.var(ddof=1))
    if len(w) == 0 or float(w.sum()) == 0:
        return np.nan
    mean = np.average(x, weights=w)
    return float(np.average((x - mean) ** 2, weights=w))


def _smd(treated: pd.Series, control: pd.Series, wt: pd.Series | None, wc: pd.Series | None) -> float:
    if wt is not None:
        t_keep = treated.notna() & wt.notna()
        treated = treated.loc[t_keep]
        wt = wt.loc[t_keep]
    else:
        treated = treated.dropna()
    if wc is not None:
        c_keep = control.notna() & wc.notna()
        control = control.loc[c_keep]
        wc = wc.loc[c_keep]
    else:
        control = control.dropna()
    if len(treated) == 0 or len(control) == 0:
        return np.nan
    mt = _weighted_mean(treated, wt)
    mc = _weighted_mean(control, wc)
    vt = _weighted_var(treated, wt)
    vc = _weighted_var(control, wc)
    denom = np.sqrt((vt + vc) / 2)
    return 0.0 if denom == 0 or np.isnan(denom) else float((mt - mc) / denom)


def balance_table(
    result_or_data: MatchResult | pd.DataFrame,
    treatment: str | None = None,
    covariates: list[str] | None = None,
    weights: str = "weights",
) -> pd.DataFrame:
    """Compute standardized mean differences before and after matching."""
    if isinstance(result_or_data, MatchResult):
        result = result_or_data
        data = result.data
        treatment = result.treatment
        covariates = result.covariates
    else:
        data = result_or_data
        if treatment is None or covariates is None:
            raise ValueError("treatment and covariates are required when passing a DataFrame.")

    x_all = _as_numeric_frame(data, covariates)
    rows = []

    for col in x_all.columns:
        t_all = x_all.loc[data[treatment] == 1, col]
        c_all = x_all.loc[data[treatment] == 0, col]
        smd_pre = _smd(t_all, c_all, None, None)

        smd_post = np.nan
        if weights in data.columns:
            matched = data[data[weights] > 0]
            x_m = _as_numeric_frame(matched, covariates)
            if col in x_m.columns:
                t_mask = matched[treatment] == 1
                c_mask = matched[treatment] == 0
                smd_post = _smd(
                    x_m.loc[t_mask, col],
                    x_m.loc[c_mask, col],
                    matched.loc[t_mask, weights],
                    matched.loc[c_mask, weights],
                )

        rows.append(
            {
                "covariate": col,
                "smd_before": smd_pre,
                "smd_after": smd_post,
                "abs_smd_before": abs(smd_pre),
                "abs_smd_after": abs(smd_post) if not pd.isna(smd_post) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("abs_smd_before", ascending=False).reset_index(drop=True)


def love_plot(
    balance: MatchResult | pd.DataFrame,
    *,
    threshold: float = 0.10,
    title: str = "Love plot",
    figsize: tuple[int, int] = (8, 6),
):
    """Create a Love plot from a MatchResult or balance table."""
    if isinstance(balance, MatchResult):
        bal = balance_table(balance)
    else:
        bal = balance.copy()

    bal = bal.sort_values("abs_smd_before", ascending=True)
    y = np.arange(len(bal))

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(bal["abs_smd_before"], y, label="Before matching", marker="o", color="red")
    if "abs_smd_after" in bal.columns:
        ax.scatter(bal["abs_smd_after"], y, label="After matching", marker="o", color="blue")
        for pos, (_, row) in enumerate(bal.iterrows()):
            ax.plot([row["abs_smd_before"], row["abs_smd_after"]], [pos, pos], alpha=0.4)

    ax.axvline(threshold, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(bal["covariate"])
    ax.set_xlabel("Absolute standardized mean difference")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


@dataclass
class XBalanceResult:
    """Container returned by :func:`xbalance`.

    This is a Python analogue of RItools::xBalance for teaching use. It tests
    pre-matching covariate balance between treatment groups under the null of
    no difference in covariate means.
    """

    formula: str | None
    treatment: str
    table: pd.DataFrame
    overall: pd.DataFrame
    report: str = "all"

    def __repr__(self) -> str:
        cols = ["covariate", "mean_treated", "mean_control", "difference", "std_diff", "t_stat", "p_value"]
        shown = self.table[cols].copy()
        with pd.option_context("display.max_rows", 50, "display.width", 120, "display.float_format", "{:.4f}".format):
            body = shown.to_string(index=False)
            overall = self.overall.to_string(index=False)
        return (
            "<XBalanceResult: tests of pre-matching balance>\n"
            f"Treatment: {self.treatment}\n"
            "Null hypothesis: no difference in covariate means between treatment groups\n\n"
            f"{body}\n\n"
            "Overall test:\n"
            f"{overall}"
        )

    def summary(self) -> pd.DataFrame:
        """Return the covariate-level balance table."""
        return self.table.copy()

    def plot(self, *, threshold: float | None = None, figsize: tuple[int, int] = (8, 6), title: str | None = None):
        """Plot signed standardized differences from xbalance()."""
        return plot_xbalance(self, threshold=threshold, figsize=figsize, title=title)


def _design_from_formula(formula: str, data: pd.DataFrame, na_rm: bool):
    try:
        import patsy
    except ImportError as exc:
        raise ImportError("xbalance(formula=...) requires patsy, installed with statsmodels.") from exc

    na_action = "drop" if na_rm else "raise"
    y, X = patsy.dmatrices(formula, data=data, return_type="dataframe", NA_action=na_action)
    if X.shape[1] and "Intercept" in X.columns:
        X = X.drop(columns=["Intercept"])
    treatment = y.columns[0]
    yy = y.iloc[:, 0]
    return treatment, yy, X


def _design_from_covariates(data: pd.DataFrame, treatment: str, covariates: list[str], na_rm: bool):
    cols = [treatment] + list(covariates)
    d = data[cols].copy()
    if na_rm:
        d = d.dropna()
    elif d.isna().any().any():
        raise ValueError("Missing values detected. Use na_rm=True to drop rows with missing treatment/covariates.")
    y = d[treatment]
    X = pd.get_dummies(d[covariates], drop_first=False, dtype=float)
    return treatment, y, X


def _validate_treatment_series(y: pd.Series, treatment: str) -> pd.Series:
    values = set(pd.Series(y).dropna().unique())
    if not values.issubset({0, 1, False, True}):
        raise ValueError(f"Treatment '{treatment}' must be binary coded 0/1 or True/False.")
    return pd.Series(y).astype(int)


def _safe_ttest(treated: pd.Series, control: pd.Series) -> tuple[float, float]:
    if len(treated) < 2 or len(control) < 2:
        return np.nan, np.nan
    if treated.var(ddof=1) == 0 and control.var(ddof=1) == 0:
        return 0.0, 1.0 if treated.mean() == control.mean() else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.ttest_ind(treated, control, equal_var=False, nan_policy="omit")
    return float(res.statistic), float(res.pvalue)


def _overall_ols_f_test(y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """Joint test that covariates do not predict treatment status."""
    try:
        import statsmodels.api as sm
        XX = sm.add_constant(X.astype(float), has_constant="add")
        model = sm.OLS(y.astype(float), XX).fit()
        if X.shape[1] == 0:
            return pd.DataFrame([{"test": "overall", "statistic": np.nan, "df_num": 0, "df_denom": np.nan, "p_value": np.nan}])
        R = np.zeros((X.shape[1], XX.shape[1]))
        R[:, 1:] = np.eye(X.shape[1])
        ftest = model.f_test(R)
        return pd.DataFrame([
            {
                "test": "overall linear balance test",
                "statistic": float(np.asarray(ftest.fvalue).ravel()[0]),
                "df_num": int(ftest.df_num),
                "df_denom": float(ftest.df_denom),
                "p_value": float(np.asarray(ftest.pvalue).ravel()[0]),
            }
        ])
    except Exception:
        return pd.DataFrame([{"test": "overall linear balance test", "statistic": np.nan, "df_num": np.nan, "df_denom": np.nan, "p_value": np.nan}])


def xbalance(
    formula: str | None = None,
    *,
    data: pd.DataFrame,
    treatment: str | None = None,
    covariates: list[str] | None = None,
    report: str = "all",
    na_rm: bool = False,
) -> XBalanceResult:
    """RItools-style pre-matching balance test.

    Parameters
    ----------
    formula:
        Patsy/R-style formula, e.g. ``"treated ~ age + C(region) + income"``.
        Categorical variables should be wrapped in ``C()`` or stored as pandas
        ``category``/string variables when using ``covariates=``.
    data:
        DataFrame containing treatment and covariates.
    treatment, covariates:
        Alternative to ``formula``. Use ``treatment="treated"`` and
        ``covariates=["age", "region"]``.
    report:
        Accepted for RItools compatibility. Currently ``"all"`` returns all
        implemented columns.
    na_rm:
        If True, drop rows with missing treatment/covariate values. If False,
        raise an error when missing values are present.

    Returns
    -------
    XBalanceResult
        Printable object with covariate-level Welch t-tests, standardized mean
        differences, and an overall linear balance test.
    """
    if formula is None and (treatment is None or covariates is None):
        raise ValueError("Provide either formula='treat ~ x1 + x2' or both treatment= and covariates=.")
    if formula is not None:
        treatment_name, y, X = _design_from_formula(formula, data, na_rm)
    else:
        treatment_name, y, X = _design_from_covariates(data, treatment, covariates, na_rm)

    y = _validate_treatment_series(y, treatment_name)
    X = X.loc[y.index].copy()

    rows = []
    for col in X.columns:
        x = pd.to_numeric(X[col], errors="coerce")
        treated = x.loc[y == 1].dropna()
        control = x.loc[y == 0].dropna()
        mean_t = float(treated.mean()) if len(treated) else np.nan
        mean_c = float(control.mean()) if len(control) else np.nan
        diff = mean_t - mean_c
        sd_pool = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2) if len(treated) > 1 and len(control) > 1 else np.nan
        std_diff = diff / sd_pool if sd_pool and not np.isnan(sd_pool) else np.nan
        t_stat, p_value = _safe_ttest(treated, control)
        rows.append(
            {
                "covariate": col,
                "n_treated": int(len(treated)),
                "n_control": int(len(control)),
                "mean_treated": mean_t,
                "mean_control": mean_c,
                "difference": diff,
                "std_diff": float(std_diff) if not pd.isna(std_diff) else np.nan,
                "abs_std_diff": abs(float(std_diff)) if not pd.isna(std_diff) else np.nan,
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )

    table = pd.DataFrame(rows).sort_values("abs_std_diff", ascending=False, na_position="last").reset_index(drop=True)
    overall = _overall_ols_f_test(y, X)
    return XBalanceResult(formula=formula, treatment=treatment_name, table=table, overall=overall, report=report)


def plot_xbalance(
    balance: XBalanceResult,
    *,
    threshold: float | None = None,
    figsize: tuple[int, int] = (8, 6),
    title: str | None = None,
):
    """Plot standardized differences from :func:`xbalance`."""
    bal = balance.table.sort_values("abs_std_diff", ascending=True)
    y = np.arange(len(bal))
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(bal["std_diff"], y, marker="s", color="black")
    ax.axvline(0, color="black", linewidth=0.8)
    if threshold is not None:
        ax.axvline(threshold, linestyle="--", linewidth=0.8, color="gray")
        ax.axvline(-threshold, linestyle="--", linewidth=0.8, color="gray")
    ax.set_yticks(y)
    ax.set_yticklabels(bal["covariate"])
    ax.set_xlabel("Standardized Differences")
    if title:
        ax.set_title(title)
    x = bal["std_diff"].dropna()
    max_abs = float(np.nanmax(np.abs(x))) if len(x) else 0.1
    limit = max(0.1, max_abs * 1.15)
    ax.set_xlim(-limit, limit)
    fig.tight_layout()
    return fig, ax
