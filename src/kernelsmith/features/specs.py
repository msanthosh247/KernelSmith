"""Feature signatures.

A CallFactory is the single source of truth for a feature's types: every
backend's implementation is checked against these, and no implementation
declares its types twice.
"""
from __future__ import annotations

from kernelsmith.dsl.graph import CallFactory
from kernelsmith.dsl.types import F4, I4

sma = CallFactory(
    "sma",
    input_signature=[F4[:], I4],
    buffer_signature=[],
    output_signature=[F4[:]],
)

ema = CallFactory(
    "ema",
    input_signature=[F4[:], I4],
    buffer_signature=[],
    output_signature=[F4[:]],
)

rolling_min_max = CallFactory(
    "rolling_min_max",
    input_signature=[F4[:], I4],
    buffer_signature=[],
    output_signature=[F4[:], F4[:]],
)

__all__ = ["ema", "rolling_min_max", "sma"]
