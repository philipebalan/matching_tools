"""
matchingtools
=============

A teaching-oriented Python library for causal matching workflows.
"""

__version__ = "0.3.0"

from .core import MatchResult, matchit, match_data, estimate_att, lm_robust
from .balance import balance_table, love_plot, XBalanceResult, xbalance, plot_xbalance
from .diagnostics import density_plot, histogram_plot, jitter_plot, plot, qq_plot, summary
from .inference import t_test, difference_in_means
from .display import screenreg
from .io import load_data, freq_table

__all__ = [
    "MatchResult",
    "matchit",
    "match_data",
    "estimate_att",
    "lm_robust",
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
    "t_test",
    "difference_in_means",
    "screenreg",
    "load_data",
    "freq_table",
]
