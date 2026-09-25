"""Soft outputs: per shot, q = the method's probability that its own prediction is wrong (spec section 8.3)."""
from .belief import BeliefMatchingGap
from .gap import LogicalGap, PreconditionError, gap_dem, q_from_gap, sample_gap

__all__ = ["BeliefMatchingGap", "LogicalGap", "PreconditionError", "gap_dem", "q_from_gap", "sample_gap"]
