"""Exact posterior under a DEM for small instances (<= 26 bits), the reference soft output (spec section 8.3)."""
from .posterior import MAX_BITS, ExactPosterior, mechanisms

__all__ = ["MAX_BITS", "ExactPosterior", "mechanisms"]
