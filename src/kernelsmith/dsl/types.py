"""Type system for the kernelsmith DSL.

One promotion table instead of a class hierarchy: operators look up their
result dtype here, and invalid operand combinations fail at graph-build time
with a message that names the offending dtypes.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from kernelsmith.errors import DslTypeError


class DType(Enum):
    BOOL = "bool"
    INT32 = "int32"
    FLOAT32 = "float32"


class Shape(Enum):
    VECTOR = "vector"
    SCALAR = "scalar"


class VarRole(Enum):
    INPUT = "input"
    PARAM = "param"
    TEMP = "temp"
    CONST = "const"


ARITH_OPS = {"+", "-", "*", "/"}
COMPARE_OPS = {">", "<", ">=", "<=", "=="}
LOGIC_OPS = {"&", "|", "^"}
UNARY_OPS = {"~", "neg"}
ALL_OPS = ARITH_OPS | COMPARE_OPS | LOGIC_OPS | UNARY_OPS

_RANK = {DType.BOOL: 0, DType.INT32: 1, DType.FLOAT32: 2}
_NUMERIC = {DType.INT32, DType.FLOAT32}

CONST_OPERAND_TYPES = (bool, int, float, np.bool_, np.integer, np.floating)


def _promote(a: DType, b: DType) -> DType:
    return a if _RANK[a] > _RANK[b] else b


def promote_dtype(a: DType, b: Optional[DType], operation: str) -> DType:
    """Result dtype of ``operation`` applied to operands of dtype ``a`` (and ``b``).

    Raises DslTypeError when the operand dtypes are not valid for the operation.
    ``b`` is None for unary operations.
    """
    if operation in UNARY_OPS:
        if operation == "neg":
            if a not in _NUMERIC:
                raise DslTypeError(f"'-' requires a numeric operand, got {a.value}")
            return a
        # "~"
        if a is not DType.BOOL:
            raise DslTypeError(
                f"'~' requires a bool operand, got {a.value} - did you mean a comparison?"
            )
        return DType.BOOL

    if operation in ARITH_OPS:
        if a not in _NUMERIC or b not in _NUMERIC:
            raise DslTypeError(
                f"'{operation}' requires numeric operands, got {a.value} and {b.value}"
            )
        # true division always promotes to float, numpy-style
        return DType.FLOAT32 if operation == "/" else _promote(a, b)

    if operation in COMPARE_OPS:
        if a not in _NUMERIC or b not in _NUMERIC:
            raise DslTypeError(
                f"'{operation}' requires numeric operands, got {a.value} and {b.value}"
            )
        return DType.BOOL

    if operation in LOGIC_OPS:
        if (a is not DType.BOOL) or (b is not DType.BOOL):
            raise DslTypeError(
                f"'{operation}' requires bool operands, got {a.value} and {b.value}"
                " - did you mean a comparison?"
            )
        return DType.BOOL

    raise DslTypeError(f"unknown operation '{operation}'")


def infer_dtype(val) -> DType:
    """DType of a python or numpy constant. bool first: python bool subclasses int."""
    if isinstance(val, (bool, np.bool_)):
        return DType.BOOL
    if isinstance(val, (int, np.integer)):
        return DType.INT32
    if isinstance(val, (float, np.floating)):
        return DType.FLOAT32
    raise DslTypeError(f"cannot infer a DType for constant of type {type(val).__name__}")


def result_shape(a: Shape, b: Optional[Shape] = None) -> Shape:
    """Scalar only when both operands are scalar - vectors broadcast."""
    if b is None:
        return a
    if (a is Shape.SCALAR) and (b is Shape.SCALAR):
        return Shape.SCALAR
    return Shape.VECTOR


class ValueSignature:
    """dtype + shape pair describing one argument or output of a feature."""

    def __init__(self, dtype: DType, shape: Shape):
        self.dtype = dtype
        self.shape = shape

    def __repr__(self):
        return f"<{self.shape.value} {self.dtype.value}>"


class DTypeSignature:
    def __init__(self, dtype: DType):
        self.dtype = dtype


class ShapeSignature:
    """Composable signature builder: VEC(F4) -> vector float32, SCAL(I4) -> scalar int32."""

    def __init__(self, shape: Shape):
        self.shape = shape

    def __call__(self, type_sig: DTypeSignature) -> ValueSignature:
        return ValueSignature(type_sig.dtype, self.shape)


F4 = DTypeSignature(DType.FLOAT32)
I4 = DTypeSignature(DType.INT32)
B1 = DTypeSignature(DType.BOOL)

VEC = ShapeSignature(Shape.VECTOR)
SCAL = ShapeSignature(Shape.SCALAR)
