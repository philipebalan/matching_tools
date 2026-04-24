"""
matchingtools
=============

A teaching-oriented Python library for causal matching workflows.
"""

__version__ = "0.2.5"

from .core import MatchResult, matchit, match_data, estimate_att
from .balance import balance_table, love_plot, XBalanceResult, xbalance, plot_xbalance
from .diagnostics import density_plot, histogram_plot, jitter_plot, plot, qq_plot, summary
from .io import load_data

__all__ = [
    "MatchResult",
    "matchit",
    "match_data",
    "estimate_att",
    "balance_table",
    "love_plot",
    "XBalanceResult",
    "xbalance",
    "plot_xbalance",
    "density_plot",
    "histogram_plot",
    "jitter_plot",
    "plot",
    "qq_plot",
    "summary",
    "load_data",
]
