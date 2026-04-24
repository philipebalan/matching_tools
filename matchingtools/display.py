from __future__ import annotations

import numpy as np


def screenreg(
    models,
    *,
    custom_model_names: list[str] | None = None,
    stars: dict | None = None,
    digits: int = 2,
    ci: bool = True,
) -> str:
    """Print a side-by-side regression table, analogous to texreg::screenreg()."""
    if stars is None:
        stars = {0.001: "***", 0.01: "**", 0.05: "*"}
    if not isinstance(models, (list, tuple)):
        models = [models]

    names = custom_model_names or [f"Model {i + 1}" for i in range(len(models))]
    if len(names) != len(models):
        raise ValueError("custom_model_names length must match number of models.")

    all_terms = []
    for model in models:
        for term in model.params.index:
            if term not in all_terms:
                all_terms.append(term)

    def star(p_value: float) -> str:
        for threshold, symbol in sorted(stars.items()):
            if p_value < threshold:
                return symbol
        return " "

    fmt = f"{{:.{digits}f}}"
    rows_coef = {}
    rows_sub = {}
    for term in all_terms:
        rows_coef[term] = []
        rows_sub[term] = []
        for model in models:
            if term not in model.params:
                rows_coef[term].append("")
                rows_sub[term].append("")
                continue
            coef = model.params[term]
            p_value = model.pvalues[term]
            rows_coef[term].append(fmt.format(coef) + star(p_value))
            if ci:
                low, high = model.conf_int().loc[term]
                rows_sub[term].append(f"[{fmt.format(low)}; {fmt.format(high)}]")
            else:
                rows_sub[term].append(f"({fmt.format(model.bse[term])})")

    footer_keys = ["R2", "Adj. R2", "N", "RMSE"]
    footer = {key: [] for key in footer_keys}
    for model in models:
        footer["R2"].append(fmt.format(getattr(model, "rsquared", np.nan)))
        footer["Adj. R2"].append(fmt.format(getattr(model, "rsquared_adj", np.nan)))
        footer["N"].append(str(int(model.nobs)))
        rmse = np.sqrt(model.mse_resid) if hasattr(model, "mse_resid") else np.nan
        footer["RMSE"].append(fmt.format(rmse))

    col_width = max(
        max(len(name) for name in names),
        max(
            max((len(value) for value in row), default=0)
            for row in list(rows_coef.values()) + list(rows_sub.values())
        ),
        8,
    ) + 2
    term_width = max(len(term) for term in all_terms + footer_keys) + 2
    sep = "=" * (term_width + col_width * len(models))
    thin_sep = "-" * (term_width + col_width * len(models))

    lines = [sep, " " * term_width + "".join(name.center(col_width) for name in names), thin_sep]
    for term in all_terms:
        lines.append(term.ljust(term_width) + "".join(value.center(col_width) for value in rows_coef[term]))
        lines.append(" " * term_width + "".join(value.center(col_width) for value in rows_sub[term]))

    lines.append(thin_sep)
    for key in footer_keys:
        lines.append(key.ljust(term_width) + "".join(value.center(col_width) for value in footer[key]))
    lines.append(sep)
    lines.append("* " + "; ".join(f"p<{threshold}: {symbol}" for threshold, symbol in sorted(stars.items())))

    table = "\n".join(lines)
    print(table)
    return table
