from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matchingtools as mt


def _distance_df():
    return pd.DataFrame(
        {
            "treated": [1, 1, 0, 0, 0, 0],
            "x": [0.10, 0.90, 0.11, 0.89, 0.40, 0.60],
            "ps": [0.10, 0.90, 0.11, 0.89, 0.40, 0.60],
            "y": [1, 2, 0, 1, 0, 1],
        },
        index=["t1", "t2", "c1", "c2", "c3", "c4"],
    )


def _plot_df():
    return pd.DataFrame(
        {
            "treated": [1, 1, 1, 0, 0, 0, 0, 0],
            "x": [0.10, 0.25, 0.80, 0.11, 0.24, 0.78, 0.55, 0.95],
            "region": ["A", "B", "C", "A", "B", "C", "A", "C"],
            "gender": ["F", "M", "F", "F", "M", "M", "F", "M"],
            "ps": [0.10, 0.25, 0.80, 0.11, 0.24, 0.78, 0.55, 0.95],
        }
    )


def test_nearest_matching_runs():
    df = _distance_df().iloc[:4].copy()
    m = mt.matchit(df, treatment="treated", covariates=["x"], method="nearest", distance="ps")
    dm = mt.match_data(m)
    assert len(dm) == 4
    assert "weights" in dm.columns


def test_nearest_ratio_two_att_weights_with_replacement():
    df = _distance_df().iloc[:4].copy()
    m = mt.matchit(df, treatment="treated", covariates=["x"], method="nearest", distance="ps", ratio=2, replace=True)
    assert m.data.loc["t1", "weights"] == 1.0
    assert m.data.loc["t2", "weights"] == 1.0
    assert m.data.loc["c1", "weights"] == 1.0
    assert m.data.loc["c2", "weights"] == 1.0


def test_nearest_no_replacement_ratio_two_weights():
    df = _distance_df()
    m = mt.matchit(df, treatment="treated", covariates=["x"], method="nearest", distance="ps", ratio=2, replace=False)
    assert (m.data.loc[["t1", "t2"], "weights"] == 1.0).all()
    assert np.isclose(m.data.loc[["c1", "c2", "c3", "c4"], "weights"].sum(), 2.0)


def test_caliper_partial_and_no_matches_have_stable_empty_pairs():
    df = _distance_df().iloc[:4].copy()
    df.loc["c2", "ps"] = 0.70
    partial = mt.matchit(df, treatment="treated", covariates=["x"], method="nearest", distance="ps", caliper=0.02)
    assert len(partial.pairs) == 1

    none = mt.matchit(df, treatment="treated", covariates=["x"], method="nearest", distance="ps", caliper=0.001)
    assert list(none.pairs.columns) == ["treated_index", "control_index", "distance"]
    assert none.pairs.empty
    assert (none.data["weights"] == 0).all()


def test_mahalanobis_numeric_uses_inverse_covariance_distance():
    df = pd.DataFrame(
        {
            "treated": [1, 1, 0, 0],
            "x1": [0.0, 3.0, 0.0, 2.0],
            "x2": [0.0, 0.0, 2.0, 1.0],
        },
        index=["t1", "t2", "c1", "c2"],
    )
    m = mt.matchit(df, treatment="treated", covariates=["x1", "x2"], method="mahalanobis")
    x = df[["x1", "x2"]].to_numpy(dtype=float)
    vi = np.linalg.pinv(np.cov(x - x.mean(axis=0), rowvar=False))
    expected = np.sqrt((x[0] - x[2]) @ vi @ (x[0] - x[2]).T)
    actual = m.pairs.query("treated_index == 't1' and control_index == 'c1'")["distance"].iloc[0]
    assert np.isclose(actual, expected)


def test_mahalanobis_handles_categorical_collinearity_with_pinv():
    df = pd.DataFrame(
        {
            "treated": [1, 1, 0, 0],
            "x": [0.0, 1.0, 0.1, 1.1],
            "region": ["A", "B", "A", "B"],
            "region_copy": ["A", "B", "A", "B"],
        }
    )
    m = mt.matchit(df, treatment="treated", covariates=["x", "region", "region_copy"], method="mahalanobis")
    assert len(m.pairs) == 2


def test_balance_table_handles_missing_values():
    df = pd.DataFrame(
        {
            "treated": [1, 1, 0, 0],
            "x": [1.0, np.nan, 1.2, 1.4],
            "weights": [1.0, 1.0, 1.0, 0.0],
        }
    )
    bal = mt.balance_table(df, treatment="treated", covariates=["x"])
    assert list(bal["covariate"]) == ["x"]
    assert "smd_after" in bal.columns


def test_xbalance_plot_is_signed_centered_and_unlabeled():
    df = pd.DataFrame(
        {
            "treated": [1, 1, 0, 0],
            "x": [1.0, 1.2, 1.8, 2.0],
            "z": [0.0, 1.0, 0.0, 0.0],
        }
    )
    bal = mt.xbalance(data=df, treatment="treated", covariates=["x", "z"])
    _, ax = bal.plot()
    xmin, xmax = ax.get_xlim()
    assert ax.get_title() == ""
    assert ax.get_legend() is None
    assert xmin < 0 < xmax
    assert np.isclose(abs(xmin), abs(xmax))
    assert ax.get_xlabel() == "Standardized Differences"


def test_plot_after_shapes():
    df = _distance_df().iloc[:4].copy()
    m = mt.matchit(df, treatment="treated", covariates=["x"], method="nearest", distance="ps")
    _, axes_default = mt.density_plot(m, variables=["x"])
    _, axes_before = mt.qq_plot(m, variables=["x"], after=False)
    _, axes_after = mt.histogram_plot(m, variables=["x"], after=True)
    assert np.asarray(axes_default).size == 2
    assert np.asarray(axes_before).size == 1
    assert np.asarray(axes_after).size == 2


def test_plot_density_defaults_to_one_figure_per_covariate():
    df = _plot_df()
    m = mt.matchit(df, treatment="treated", covariates=["x", "region"], method="nearest", distance="ps")
    out = mt.plot(m, "density", interactive=False)
    assert isinstance(out, list)
    assert len(out) == 2
    for _, axes in out:
        assert np.asarray(axes).size == 2


def test_categorical_density_uses_bar_charts_in_both_panels():
    df = _plot_df()
    m = mt.matchit(df, treatment="treated", covariates=["x", "region"], method="nearest", distance="ps")
    _, axes = mt.plot(m, "density", variables=["region"], interactive=False)
    for ax in np.asarray(axes).ravel():
        assert len(ax.containers) >= 2


def test_categorical_qq_expands_levels_and_draws_reference_lines():
    df = _plot_df()
    m = mt.matchit(df, treatment="treated", covariates=["x", "region"], method="nearest", distance="ps")
    _, axes = mt.plot(m, "qq", variables=["region"], interactive=False)
    axes_array = np.asarray(axes)
    assert axes_array.shape == (3, 2)
    for ax in axes_array.ravel():
        assert len(ax.lines) >= 3


def test_binary_categorical_qq_uses_one_positive_level_row():
    df = _plot_df()
    m = mt.matchit(df, treatment="treated", covariates=["gender"], method="nearest", distance="ps")
    _, axes = mt.plot(m, "qq", variables=["gender"], interactive=False)
    assert np.asarray(axes).shape == (1, 2)


def test_plot_variables_subset_and_interactive_argument():
    df = _plot_df()
    m = mt.matchit(df, treatment="treated", covariates=["x", "region"], method="nearest", distance="ps")
    _, axes = mt.plot(m, "density", variables=["region"], interactive=False)
    assert np.asarray(axes).size == 2


def test_readme_version_matches_package():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert f"{mt.__version__}" in text
    assert "0.2.2" not in text


def test_validation_errors_are_clear():
    df = _distance_df()
    with pytest.raises(ValueError, match="caliper"):
        mt.matchit(df, treatment="treated", covariates=["x"], method="nearest", distance="ps", caliper=-1)
    with pytest.raises(ValueError, match="column"):
        mt.estimate_att(df, outcome="missing", treatment="treated")
