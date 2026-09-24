"""Calibration and selective-risk metrics (spec section 8.5). Post-hoc calibrators come in M4."""
from .metrics import brier, ece, error_vs_discard, nll, reliability

__all__ = ["brier", "ece", "error_vs_discard", "nll", "reliability"]
