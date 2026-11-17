"""Diagnostics on the data itself (plan A2: shot order and independence)."""
from .order import autocorr, block_heterogeneity, block_variance_ratio, ljung_box, runs_test, trend

__all__ = ["autocorr", "block_heterogeneity", "block_variance_ratio", "ljung_box", "runs_test", "trend"]
