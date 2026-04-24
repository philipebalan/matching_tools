from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

Method = Literal["nearest", "optimal", "mahalanobis"]
Estimand = Literal["ATT"]
PAIR_COLUMNS = ["treated_index", "control_index", "distance"]


@dataclass
class MatchResult:
    """Container returned by :func:`matchit`.

    Attributes
    ----------
    data:
        Original data with propensity score and matching weights added.
    treatment:
        Treatment column name.
    covariates:
        Covariates used for propensity score estimation or Mahalanobis distance.
    method:
        Matching method used.
    estimand:
        Causal estimand. Currently ATT only.
    pairs:
        DataFrame with treated index, control index, and matching distance.
    ps_model:
        Fitted statsmodels logit object when propensity scores were estimated.
    """

    data: pd.DataFrame
    treatment: str
    covariates: list[str]
    method: str
    estimand: str
    pairs: pd.DataFrame
    ps_model: object | None = None

    def matched_data(self) -> pd.DataFrame:
        """Return the matched dataset with positive matching weights."""
        return match_data(self)

    def __repr__(self) -> str:
        d = self.data
        matched = d[d["weights"] > 0]
        return (
            "<MatchResult>\n"
            f"Method: {self.method} | Estimand: {self.estimand}\n"
            f"Treatment: {self.treatment}\n"
            f"Original sample: {len(d)} obs "
            f"({int((d[self.treatment] == 1).sum())} treated, {int((d[self.treatment] == 0).sum())} control)\n"
            f"Matched sample: {len(matched)} obs "
            f"({int((matched[self.treatment] == 1).sum())} treated, {int((matched[self.treatment] == 0).sum())} control)\n"
            f"Matched pairs: {len(self.pairs)}"
        )

    def summary(self, standardize: bool = True) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Return a MatchIt-style summary.

        If standardize=False, return only the sample-size overview.
        If standardize=True, return overview, balance, and pairs tables.
        """
        d = self.data
        matched = d[d["weights"] > 0]
        overview = pd.DataFrame(
            {
                "method": [self.method],
                "estimand": [self.estimand],
                "original_n": [len(d)],
                "original_treated": [int((d[self.treatment] == 1).sum())],
                "original_control": [int((d[self.treatment] == 0).sum())],
                "matched_n": [len(matched)],
                "matched_treated": [int((matched[self.treatment] == 1).sum())],
                "matched_control": [int((matched[self.treatment] == 0).sum())],
                "matched_pairs": [len(self.pairs)],
                "control_weight_sum": [matched.loc[matched[self.treatment] == 0, "weights"].sum()],
            }
        )
        if not standardize:
            return overview

        from .balance import balance_table

        return {
            "overview": overview,
            "balance": balance_table(self),
            "pairs": self.pairs.copy(),
        }


def _validate_binary_treatment(df: pd.DataFrame, treatment: str) -> None:
    if treatment not in df.columns:
        raise ValueError(f"Treatment column '{treatment}' not found in data.")
    values = set(df[treatment].dropna().unique())
    if not values.issubset({0, 1, False, True}):
        raise ValueError(f"Treatment column '{treatment}' must be binary coded 0/1.")


def _validate_columns(df: pd.DataFrame, columns: Iterable[str], *, role: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{role} column(s) not found in data: {missing}")


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(columns=PAIR_COLUMNS)


def _formula_from_covariates(treatment: str, covariates: Iterable[str]) -> str:
    return f"{treatment} ~ " + " + ".join(covariates)


def _estimate_propensity_scores(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    formula: str | None = None,
) -> tuple[pd.Series, object]:
    formula = formula or _formula_from_covariates(treatment, covariates)
    try:
        model = smf.logit(formula, data=df).fit(disp=False)
        ps = pd.Series(model.predict(df), index=df.index, name="distance")
    except Exception as exc:
        raise ValueError(
            "Could not estimate propensity scores. Common causes include missing "
            "values, perfect separation, invalid formula terms, or a treatment "
            "column that is not binary coded 0/1."
        ) from exc
    return ps, model


def _nearest_pairs(
    df: pd.DataFrame,
    treatment: str,
    distance: str,
    ratio: int,
    replace: bool,
    caliper: float | None,
) -> pd.DataFrame:
    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]

    if treated.empty or control.empty:
        raise ValueError("Both treated and control groups must contain observations.")

    available_controls = list(control.index)
    pairs = []

    for t_idx, t_row in treated.iterrows():
        if not available_controls:
            break

        ctrl = df.loc[available_controls]
        distances = (ctrl[distance] - t_row[distance]).abs().sort_values()
        if caliper is not None:
            distances = distances[distances <= caliper]
        selected = list(distances.head(ratio).index)

        for c_idx in selected:
            pairs.append(
                {
                    "treated_index": t_idx,
                    "control_index": c_idx,
                    "distance": float(abs(df.loc[t_idx, distance] - df.loc[c_idx, distance])),
                }
            )

        if not replace:
            available_controls = [idx for idx in available_controls if idx not in selected]

    return pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def _optimal_pairs(
    df: pd.DataFrame,
    treatment: str,
    distance: str,
    ratio: int,
    caliper: float | None,
) -> pd.DataFrame:
    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]

    if ratio != 1:
        raise NotImplementedError("Optimal matching currently supports ratio=1 only.")
    if len(control) < len(treated):
        raise ValueError("Optimal 1:1 ATT matching requires at least as many controls as treated units.")

    cost = np.abs(treated[distance].to_numpy()[:, None] - control[distance].to_numpy()[None, :])
    if caliper is not None:
        cost = np.where(cost <= caliper, cost, 1e9)

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= 1e9:
            continue
        pairs.append(
            {
                "treated_index": treated.index[r],
                "control_index": control.index[c],
                "distance": float(cost[r, c]),
            }
        )
    return pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def _mahalanobis_pairs(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    ratio: int,
    replace: bool,
    caliper: float | None,
) -> pd.DataFrame:
    work = df[covariates].copy()
    work = pd.get_dummies(work, drop_first=True, dtype=float)
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        raise ValueError("No complete covariate rows remain for Mahalanobis matching.")

    valid_df = df.loc[work.index]
    x = work.to_numpy(dtype=float)

    treated_pos = np.where(valid_df[treatment].to_numpy() == 1)[0]
    control_pos = np.where(valid_df[treatment].to_numpy() == 0)[0]
    if len(treated_pos) == 0 or len(control_pos) == 0:
        raise ValueError("Both treated and control groups must contain complete observations for Mahalanobis matching.")

    treated_index = valid_df.index[treated_pos]
    control_index = valid_df.index[control_pos]
    centered = x - x.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    cov = np.atleast_2d(cov)
    try:
        vi = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        vi = np.linalg.pinv(cov)
    dist_matrix = cdist(x[treated_pos], x[control_pos], metric="mahalanobis", VI=vi)

    available = list(range(len(control_index)))
    pairs = []
    for i, t_idx in enumerate(treated_index):
        if not available:
            break
        row = pd.Series(dist_matrix[i, available], index=available).sort_values()
        if caliper is not None:
            row = row[row <= caliper]
        selected = list(row.head(ratio).index)
        for j in selected:
            pairs.append(
                {
                    "treated_index": t_idx,
                    "control_index": control_index[j],
                    "distance": float(dist_matrix[i, j]),
                }
            )
        if not replace:
            available = [j for j in available if j not in selected]

    return pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def _weights_from_pairs(index: pd.Index, pairs: pd.DataFrame) -> pd.Series:
    weights = pd.Series(0.0, index=index, name="weights")
    if pairs.empty:
        return weights

    for treated_idx, group in pairs.groupby("treated_index", sort=False):
        weights.loc[treated_idx] = 1.0
        control_weight = 1.0 / len(group)
        for control_idx in group["control_index"]:
            weights.loc[control_idx] += control_weight

    return weights


def matchit(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    *,
    formula: str | None = None,
    method: Method = "nearest",
    estimand: Estimand = "ATT",
    ratio: int = 1,
    replace: bool = True,
    caliper: float | None = None,
    distance: str | None = None,
) -> MatchResult:
    """Match treated and control units using a MatchIt-like interface.

    Parameters
    ----------
    data:
        Input DataFrame.
    treatment:
        Binary treatment column coded 0/1.
    covariates:
        Covariates used for propensity score estimation or Mahalanobis matching.
        For categorical variables in propensity score models, use Patsy syntax in
        ``formula``, e.g. ``C(region)``.
    formula:
        Optional propensity score formula, e.g.
        ``"treated ~ age + C(gender) + hdi + I(hdi ** 2)"``.
    method:
        ``"nearest"``, ``"optimal"``, or ``"mahalanobis"``.
    estimand:
        Currently only ``"ATT"``.
    ratio:
        Number of controls per treated unit.
    replace:
        Whether controls can be reused. Used by nearest and Mahalanobis methods.
    caliper:
        Optional maximum allowed distance.
    distance:
        Existing propensity-score column. If omitted, scores are estimated.

    Returns
    -------
    MatchResult
    """
    if estimand != "ATT":
        raise NotImplementedError("Only ATT is currently implemented.")
    if ratio < 1:
        raise ValueError("ratio must be >= 1.")
    if caliper is not None and caliper < 0:
        raise ValueError("caliper must be >= 0.")

    _validate_binary_treatment(data, treatment)
    _validate_columns(data, covariates, role="Covariate")
    df = data.copy()
    ps_model = None

    if method in {"nearest", "optimal"}:
        if distance is None:
            ps, ps_model = _estimate_propensity_scores(df, treatment, covariates, formula)
            df["distance"] = ps
            distance = "distance"
        elif distance not in df.columns:
            raise ValueError(f"Distance column '{distance}' not found in data.")

        if method == "nearest":
            pairs = _nearest_pairs(df, treatment, distance, ratio, replace, caliper)
        else:
            pairs = _optimal_pairs(df, treatment, distance, ratio, caliper)

    elif method == "mahalanobis":
        pairs = _mahalanobis_pairs(df, treatment, covariates, ratio, replace, caliper)
    else:
        raise ValueError("method must be one of: 'nearest', 'optimal', 'mahalanobis'.")

    if pairs.empty:
        pairs = _empty_pairs()
    df["weights"] = _weights_from_pairs(df.index, pairs)

    return MatchResult(
        data=df,
        treatment=treatment,
        covariates=covariates,
        method=method,
        estimand=estimand,
        pairs=pairs,
        ps_model=ps_model,
    )


def match_data(result: MatchResult) -> pd.DataFrame:
    """Return matched observations with positive matching weights."""
    return result.data.loc[result.data["weights"] > 0].copy()


def estimate_att(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    *,
    weights: str = "weights",
    robust: str = "HC2",
):
    """Estimate ATT from a matched dataset using weighted OLS.

    Returns the fitted statsmodels regression object. The treatment coefficient
    is the difference in means adjusted by matching weights.
    """
    _validate_columns(data, [outcome, treatment, weights], role="Required")
    formula = f"{outcome} ~ {treatment}"
    model = smf.wls(formula, data=data, weights=data[weights]).fit(cov_type=robust)
    return model
