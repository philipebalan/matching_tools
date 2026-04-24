from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a dataset from .csv, .dta, .xlsx, or .parquet.

    Parameters
    ----------
    path:
        File path.

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".dta":
        return pd.read_stata(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def freq_table(*columns: pd.Series, normalize: bool = False) -> pd.DataFrame:
    """Cross-tabulation, analogous to R's table()."""
    if len(columns) == 1:
        result = columns[0].value_counts(normalize=normalize).sort_index().to_frame()
        result.columns = ["proportion" if normalize else "count"]
        return result
    if len(columns) == 2:
        return pd.crosstab(columns[0], columns[1], normalize=normalize)
    raise ValueError("freq_table() accepts 1 or 2 Series.")
