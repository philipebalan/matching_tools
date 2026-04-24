from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .core import MatchResult, match_data
from .balance import balance_table, XBalanceResult, plot_xbalance


PlotKind = str


def _select_variables(
    result: MatchResult,
    variables: list[str] | str | None = None,
    *,
    default: str = "covariates",
) -> list[str]:
    """Return requested variables.

    Parameters
    ----------
    result:
        MatchResult object.
    variables:
        Explicit variable name or list of names.
    default:
        ``"covariates"`` uses all matching covariates. This is the default
        used for QQ and density plots, matching R/MatchIt behavior.
        ``"distance"`` uses the propensity score/distance when available,
        falling back to covariates. This is used for histograms.
    """
    if variables is None:
        if default == "distance" and "distance" in result.data.columns:
            variables = ["distance"]
        else:
            variables = list(result.covariates)
    elif isinstance(variables, str):
        variables = [variables]

    missing = [v for v in variables if v not in result.data.columns]
    if missing:
        raise ValueError(f"Variables not found in data: {missing}")
    return list(variables)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _before_after_frames(result: MatchResult, variable: str):
    d = result.data
    md = match_data(result)
    t = result.treatment
    before_t = d.loc[d[t] == 1, variable].dropna()
    before_c = d.loc[d[t] == 0, variable].dropna()
    after_t = md.loc[md[t] == 1, variable].dropna()
    after_c = md.loc[md[t] == 0, variable].dropna()
    return before_t, before_c, after_t, after_c


def _comparison_panels(before_t: pd.Series, before_c: pd.Series, after_t: pd.Series, after_c: pd.Series, after: bool | None):
    if after is None:
        return [("All", before_t, before_c), ("Matched", after_t, after_c)]
    if after:
        return [("Matched", after_t, after_c)]
    return [("All", before_t, before_c)]


def _category_order(*series: pd.Series) -> list:
    vals = []
    for s in series:
        vals.extend(list(pd.Series(s).dropna().unique()))
    try:
        return sorted(pd.unique(vals))
    except TypeError:
        return list(pd.unique(vals))


def _plot_categorical_distribution(
    ax,
    treated: pd.Series,
    control: pd.Series,
    title: str,
    *,
    cats: list | None = None,
    ymax: float | None = None,
    ylabel="Proportion",
):
    cats = cats if cats is not None else _category_order(treated, control)
    x = np.arange(len(cats))
    width = 0.38
    t_prop = treated.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    c_prop = control.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    ax.bar(x - width / 2, c_prop.values, width, label="Control", alpha=0.75)
    ax.bar(x + width / 2, t_prop.values, width, label="Treated", alpha=0.75)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cats], rotation=35, ha="right")
    ax.legend()


def _max_category_proportion(panels: list[tuple[str, pd.Series, pd.Series]], cats: list) -> float:
    ymax = 0.0
    for _, treated, control in panels:
        t_prop = treated.value_counts(normalize=True).reindex(cats, fill_value=0.0)
        c_prop = control.value_counts(normalize=True).reindex(cats, fill_value=0.0)
        ymax = max(ymax, float(t_prop.max()), float(c_prop.max()))
    return min(1.0, ymax * 1.15 + 0.02) if ymax else 1.0


def _plot_numeric_density(ax, treated: pd.Series, control: pd.Series, title: str, xlabel: str):
    plotted = False
    if len(control) > 1 and control.nunique() > 1:
        control.plot(kind="kde", ax=ax, label="Control")
        plotted = True
    if len(treated) > 1 and treated.nunique() > 1:
        treated.plot(kind="kde", ax=ax, label="Treated")
        plotted = True
    if not plotted:
        # Fall back to a rug-style scatter when KDE is not estimable.
        if len(control):
            ax.scatter(control, np.zeros(len(control)), s=18, facecolors="none", edgecolors="black", label="Control")
        if len(treated):
            ax.scatter(treated, np.ones(len(treated)) * 0.02, s=18, facecolors="none", edgecolors="black", label="Treated")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend()


def density_plot(result: MatchResult, variables: list[str] | None = None, *, after: bool | None = None, figsize=(10, 4)):
    """Distribution plots for covariates or propensity scores.

    For numeric variables, this draws kernel densities. For categorical variables,
    it draws treated/control category proportions. The layout follows MatchIt's
    diagnostic logic: after=None compares All and Matched panels, after=False
    shows only All observations, and after=True shows only Matched observations.
    """
    variables = _select_variables(result, variables, default="covariates")
    figs_axes = []
    for var in variables:
        before_t, before_c, after_t, after_c = _before_after_frames(result, var)
        panels = _comparison_panels(before_t, before_c, after_t, after_c, after)
        fig, axes_grid = plt.subplots(1, len(panels), figsize=figsize, squeeze=False)
        axes = axes_grid.ravel()
        if _is_numeric(result.data[var]):
            for ax, (title, treated, control) in zip(axes, panels):
                _plot_numeric_density(ax, treated, control, title, var)
        else:
            cats = _category_order(before_t, before_c, after_t, after_c)
            ymax = _max_category_proportion(panels, cats)
            for ax, (title, treated, control) in zip(axes, panels):
                _plot_categorical_distribution(ax, treated, control, title, cats=cats, ymax=ymax)
        fig.suptitle(f"Density plots: {var}", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        figs_axes.append((fig, axes))
    return figs_axes[0] if len(figs_axes) == 1 else figs_axes


def _hist_panel(ax, values: pd.Series, title: str, *, bins=20, xlabel="Propensity Score"):
    if len(values):
        ax.hist(values.dropna(), bins=bins, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")


def _categorical_count_panel(ax, values: pd.Series, title: str):
    counts = values.dropna().value_counts().sort_index()
    ax.bar([str(i) for i in counts.index], counts.values, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)


def histogram_plot(result: MatchResult, variables: list[str] | None = None, *, after: bool | None = None, bins: int = 20, figsize=(10, 7)):
    """Histograms for raw/matched treated/control units.

    By default this plots propensity scores (`distance`) in four panels:
    Raw Treated, Matched Treated, Raw Control, and Matched Control.
    Categorical variables are supported by using count bar charts in the same
    layout. Use after=False for raw treated/control only or after=True for
    matched treated/control only.
    """
    variables = _select_variables(result, variables, default="distance")
    figs_axes = []
    for var in variables:
        before_t, before_c, after_t, after_c = _before_after_frames(result, var)
        if after is None:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            panels = [
                (axes[0, 0], before_t, "Raw Treated"),
                (axes[0, 1], after_t, "Matched Treated"),
                (axes[1, 0], before_c, "Raw Control"),
                (axes[1, 1], after_c, "Matched Control"),
            ]
        else:
            title_prefix = "Matched" if after else "Raw"
            treated = after_t if after else before_t
            control = after_c if after else before_c
            fig, axes_grid = plt.subplots(1, 2, figsize=figsize, squeeze=False)
            axes = axes_grid.ravel()
            panels = [
                (axes[0], treated, f"{title_prefix} Treated"),
                (axes[1], control, f"{title_prefix} Control"),
            ]
        if _is_numeric(result.data[var]):
            for ax, vals, title in panels:
                _hist_panel(ax, vals, title, bins=bins, xlabel="Propensity Score" if var == "distance" else var)
        else:
            for ax, vals, title in panels:
                _categorical_count_panel(ax, vals, title)
        fig.suptitle(f"Histograms: {var}", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        figs_axes.append((fig, axes))
    return figs_axes[0] if len(figs_axes) == 1 else figs_axes


def _qq_panel_numeric(ax, treated: pd.Series, control: pd.Series, var: str, title: str):
    n = min(len(treated), len(control))
    if n < 2:
        ax.set_title(title)
        ax.text(0.5, 0.5, "Not enough observations", transform=ax.transAxes, ha="center")
        return
    probs = np.linspace(0.01, 0.99, n)
    qt = np.quantile(treated, probs)
    qc = np.quantile(control, probs)
    ax.scatter(qc, qt, s=18, facecolors="none", edgecolors="black", linewidths=0.8)
    lo = min(np.min(qc), np.min(qt))
    hi = max(np.max(qc), np.max(qt))
    pad = 0.04 * (hi - lo) if hi > lo else 1.0
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.plot([lo, hi], [lo + pad, hi + pad], linestyle=(0, (4, 4)), linewidth=0.8)
    ax.plot([lo, hi], [lo - pad, hi - pad], linestyle=(0, (4, 4)), linewidth=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_ylabel(var)


def _dummy_levels(series: pd.Series) -> list:
    cats = _category_order(series)
    if len(cats) == 2:
        return [cats[-1]]
    return cats


def _indicator_values(series: pd.Series, level) -> np.ndarray:
    return (series == level).astype(float).to_numpy()


def _paired_binary_points(treated: pd.Series, control: pd.Series, level, *, rng: np.random.Generator, jitter: float) -> tuple[np.ndarray, np.ndarray]:
    t_vals = np.sort(_indicator_values(treated, level))
    c_vals = np.sort(_indicator_values(control, level))
    n = min(len(t_vals), len(c_vals))
    if n == 0:
        return np.array([]), np.array([])
    x = c_vals[:n] + rng.normal(0, jitter, n)
    y = t_vals[:n] + rng.normal(0, jitter, n)
    return np.clip(x, -0.08, 1.08), np.clip(y, -0.08, 1.08)


def _qq_panel_categorical_level(ax, treated: pd.Series, control: pd.Series, level, title: str, *, rng: np.random.Generator):
    x, y = _paired_binary_points(treated, control, level, rng=rng, jitter=0.025)
    if len(x):
        ax.scatter(x, y, s=18, facecolors="none", edgecolors="black", linewidths=0.8)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
    ax.plot([0, 1], [0.15, 1.15], linestyle=(0, (4, 4)), linewidth=0.8, color="black")
    ax.plot([0, 1], [-0.15, 0.85], linestyle=(0, (4, 4)), linewidth=0.8, color="black")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title(title)
    ax.set_xlabel("Control")


def qq_plot(result: MatchResult, variables: list[str] | None = None, *, after: bool | None = None, figsize=(9, 4)):
    """Empirical QQ-style plots comparing treated and control distributions.

    Numeric variables use quantile-quantile points. Categorical variables are
    expanded into binary level indicators and shown as MatchIt-style eQQ panels.
    With after=None each variable is shown in All and Matched panels. Use
    after=False or after=True to show a single panel.
    """
    variables = _select_variables(result, variables, default="covariates")
    figs_axes = []
    for var in variables:
        before_t, before_c, after_t, after_c = _before_after_frames(result, var)
        panels = _comparison_panels(before_t, before_c, after_t, after_c, after)
        if _is_numeric(result.data[var]):
            fig, axes_grid = plt.subplots(1, len(panels), figsize=figsize, sharex=False, sharey=False, squeeze=False)
            axes = axes_grid.ravel()
            for pos, (ax, (title, treated, control)) in enumerate(zip(axes, panels)):
                _qq_panel_numeric(ax, treated, control, var, title)
                if pos > 0:
                    ax.set_ylabel("")
        else:
            levels = _dummy_levels(result.data[var])
            fig_height = max(figsize[1], 2.2 * len(levels))
            fig, axes_grid = plt.subplots(len(levels), len(panels), figsize=(figsize[0], fig_height), sharex=True, sharey=True, squeeze=False)
            axes = axes_grid
            rng = np.random.default_rng(123)
            for row, level in enumerate(levels):
                for col, (title, treated, control) in enumerate(panels):
                    ax = axes_grid[row, col]
                    _qq_panel_categorical_level(ax, treated, control, level, title, rng=rng)
                    if col == 0:
                        ax.set_ylabel(f"{var}={level}\nTreated")
        fig.suptitle("eQQ Plots", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        figs_axes.append((fig, axes))
    return figs_axes[0] if len(figs_axes) == 1 else figs_axes


def jitter_plot(
    result: MatchResult,
    variable: str = "distance",
    *,
    after: bool | None = None,
    jitter: float = 0.09,
    figsize=(8, 5),
    random_state: int = 123,
):
    """MatchIt-style jitter plot for propensity scores or another variable.

    Numeric variables are placed on the x-axis directly. Categorical variables
    are placed on integer category positions with horizontal jitter.
    """
    d = result.data.copy()
    if variable not in d.columns:
        raise ValueError(f"Variable '{variable}' not found. For Mahalanobis-only matching, pass a covariate.")
    rng = np.random.default_rng(random_state)
    t = result.treatment
    matched = d["weights"] > 0
    treated = d[t] == 1
    if after is None:
        groups = [
            ("Unmatched Treated Units", treated & ~matched, 3.0),
            ("Matched Treated Units", treated & matched, 2.0),
            ("Matched Control Units", ~treated & matched, 1.0),
            ("Unmatched Control Units", ~treated & ~matched, 0.0),
        ]
    elif after:
        groups = [
            ("Matched Treated Units", treated & matched, 1.0),
            ("Matched Control Units", ~treated & matched, 0.0),
        ]
    else:
        groups = [
            ("Treated Units", treated, 1.0),
            ("Control Units", ~treated, 0.0),
        ]
    fig, ax = plt.subplots(figsize=figsize)
    numeric = _is_numeric(d[variable])
    if not numeric:
        cats = _category_order(d[variable])
        mapper = {c: i for i, c in enumerate(cats)}
    for label, mask, y0 in groups:
        vals = d.loc[mask, variable].dropna()
        if vals.empty:
            continue
        if numeric:
            x = vals.to_numpy(dtype=float)
        else:
            x = vals.map(mapper).to_numpy(dtype=float) + rng.normal(0, jitter, len(vals))
        y = y0 + rng.normal(0, jitter, len(vals))
        ax.scatter(x, y, s=24, facecolors="none", edgecolors="black", linewidths=0.8)
        ax.text(0.5, y0 + 0.28, label, transform=ax.get_yaxis_transform(), ha="center", va="center")
    ax.set_title("Distribution of Propensity Scores" if variable == "distance" else f"Jitter plot: {variable}", fontweight="bold")
    ax.set_xlabel("Propensity Score" if variable == "distance" else variable)
    ax.set_yticks([])
    y_values = [y0 for _, _, y0 in groups]
    ax.set_ylim(min(y_values) - 0.55, max(y_values) + 0.55)
    if numeric:
        xmin = d[variable].min()
        xmax = d[variable].max()
        pad = 0.04 * (xmax - xmin) if xmax > xmin else 0.05
        ax.set_xlim(xmin - pad, xmax + pad)
    else:
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([str(c) for c in cats], rotation=35, ha="right")
        ax.set_xlim(-0.6, len(cats) - 0.4)
    fig.tight_layout()
    return fig, ax


def plot(result: MatchResult | XBalanceResult, kind: PlotKind = "density", *, variables: list[str] | None = None, variable: str | None = None, interactive: bool = False, **kwargs):
    """MatchIt/RItools-style plotting wrapper.

    Examples
    --------
    plot(m, "density", variables=["age"])
    plot(m, "qq", variables=["age", "region"])
    plot(m, "jitter")
    plot(m, "histogram")

    The interactive argument is accepted for R-style compatibility but is ignored.
    """
    if isinstance(result, XBalanceResult):
        return plot_xbalance(result, **kwargs)

    kind = kind.lower()
    if kind in {"density", "dens"}:
        return density_plot(result, variables=variables, **kwargs)
    if kind in {"qq", "eqq"}:
        return qq_plot(result, variables=variables, **kwargs)
    if kind in {"jitter", "jitterplot"}:
        return jitter_plot(result, variable=variable or "distance", **kwargs)
    if kind in {"histogram", "hist", "histogram_plot"}:
        return histogram_plot(result, variables=variables, **kwargs)
    raise ValueError("kind must be one of: 'density', 'qq', 'jitter', 'histogram'.")


def summary(result: MatchResult, *, standardize: bool = True) -> dict[str, pd.DataFrame]:
    """Return a MatchIt-style summary dictionary."""
    overview = result.summary(standardize=False)
    out = {"overview": overview, "pairs": result.pairs.copy()}
    if standardize:
        out["balance"] = balance_table(result)
    return out
