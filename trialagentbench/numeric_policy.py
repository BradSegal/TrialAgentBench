"""Numerical boundary constants shared by public benchmark evaluation."""

from __future__ import annotations

from typing import Final

TRIALEVAL_FLEXIBLE_COX_PENALIZER_V1: Final = 1e-4
TRIALEVAL_FLEXIBLE_COX_STEP_SIZE_V1: Final = 0.5


__all__ = [
    "TRIALEVAL_FLEXIBLE_COX_PENALIZER_V1",
    "TRIALEVAL_FLEXIBLE_COX_STEP_SIZE_V1",
]
