"""Type system for the kernelsmith DSL.

One promotion table instead of a class hierarchy: operators look up their
result dtype here, and invalid operand combinations fail at graph-build time
with a message that names the offending dtypes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

from kernelsmith.errors import DslTypeError


class DType(Enum):
    BOOL = "bool"
    INT32 = "int32"
    FLOAT32 = "float32"

    @property
    def is_numeric(self) -> bool:
        return self in (DType.INT32, DType.FLOAT32)

    @property
    def rank(self) -> int:
        """Position in the promotion lattice: bool < int32 < float32."""
        return {DType.BOOL: 0, DType.INT32: 1, DType.FLOAT32: 2}[self]

    def join(self, other: DType) -> DType:
        """The wider of two dtypes - the least upper bound in the lattice."""
        return self if self.rank > other.rank else other

    @classmethod
    def infer_from_constant(cls, val: Any) -> DType:
        """DType of a python or numpy constant. bool first: python bool subclasses int."""
        if isinstance(val, (bool, np.bool_)):
            return cls.BOOL
        if isinstance(val, (int, np.integer)):
            return cls.INT32
        if isinstance(val, (float, np.floating)):
            return cls.FLOAT32
        raise DslTypeError(f"cannot infer a DType for constant of type {type(val).__name__}")


# kept at module level on purpose: any plain attribute placed in an Enum body
# becomes a member, so putting this inside DType would add a fourth dtype
CONST_OPERAND_TYPES = (bool, int, float, np.bool_, np.integer, np.floating)


class Shape(Enum):
    """What a value is, relative to the graph's one time axis.

    VECTOR  a series on the time axis - one element per bar, ``n_bars`` long.
    SCALAR  one value per parameter set.
    TABLE   reference data off the time axis: any length, shared by every
            parameter set, read only by features. A price history that a
            simulation resamples is a table; the simulated path is a vector.
    """

    VECTOR = "vector"
    SCALAR = "scalar"
    TABLE = "table"

    def combine(self, other: Optional[Shape] = None) -> Shape:
        """Result shape of an elementwise op.

        Unary ops keep their operand's shape; binary ops are scalar only when
        both operands are scalar - otherwise the scalar broadcasts. A table has
        no time axis to compute along, so it takes part in no elementwise op.
        """
        if self is Shape.TABLE or other is Shape.TABLE:
            raise DslTypeError("tables can only be passed to features, not used in arithmetic")
        if other is None:
            return self
        if self is Shape.SCALAR and other is Shape.SCALAR:
            return Shape.SCALAR
        return Shape.VECTOR


class VarRole(Enum):
    INPUT = "input"
    PARAM = "param"
    TEMP = "temp"
    CONST = "const"


class OpCategory(Enum):
    """Every elementwise operator the DSL knows, grouped by typing rule."""

    UNARY = frozenset({"~", "neg"})
    ARITHMETIC = frozenset({"+", "-", "*", "/"})
    COMPARE = frozenset({">", "<", ">=", "<=", "==", "!="})
    LOGIC = frozenset({"&", "|", "^"})

    @classmethod
    def classify(cls, operation: str) -> OpCategory:
        for category in cls:
            if operation in category.value:
                return category
        raise DslTypeError(f"unknown operation '{operation}'")


def result_dtype(a: DType, b: Optional[DType], operation: str) -> DType:
    """Result dtype of ``operation`` applied to operands of dtype ``a`` (and ``b``).

    ``b`` is None for unary operations. Raises DslTypeError when the operand
    dtypes - or the number of operands - are not valid for the operation.
    """
    category = OpCategory.classify(operation)

    if category is OpCategory.UNARY:
        if b is not None:
            raise DslTypeError(f"unary operation '{operation}' takes only one operand")
        if operation == "neg":
            if not a.is_numeric:
                raise DslTypeError(f"'-' requires a numeric operand, got {a.value}")
            return a
        # "~"
        if a is not DType.BOOL:
            raise DslTypeError(
                f"'~' requires a bool operand, got {a.value} - did you mean a comparison?"
            )
        return DType.BOOL

    # every remaining category is binary
    if b is None:
        raise DslTypeError(f"binary operation '{operation}' requires two operands")

    if category is OpCategory.ARITHMETIC:
        if not a.is_numeric or not b.is_numeric:
            raise DslTypeError(
                f"'{operation}' requires numeric operands, got {a.value} and {b.value}"
            )
        # true division always promotes to float, numpy-style
        return DType.FLOAT32 if operation == "/" else a.join(b)

    if category is OpCategory.COMPARE:
        if not a.is_numeric or not b.is_numeric:
            raise DslTypeError(
                f"'{operation}' requires numeric operands, got {a.value} and {b.value}"
            )
        return DType.BOOL

    if category is OpCategory.LOGIC:
        if a is not DType.BOOL or b is not DType.BOOL:
            raise DslTypeError(
                f"'{operation}' requires bool operands, got {a.value} and {b.value}"
                " - did you mean a comparison?"
            )
        return DType.BOOL

    # reachable only if a category is added above without a typing rule here;
    # failing now beats handing back a dtype of None that breaks much later
    raise DslTypeError(f"no typing rule for {category.name} operation '{operation}'")


@dataclass(frozen=True, repr=False)
class Signature:
    """dtype plus shape, describing one argument, output or scratch buffer.

    Scalar by default; subscripting with a slice gives the series form, so a
    signature reads the way the value does: ``F4`` is a float32 scalar,
    ``F4[:]`` a float32 series and ``F4.table`` a float32 table.

    Frozen because F4 / I4 / B1 are shared module-level singletons - mutating
    one would change every signature built from it - and because being
    hashable keeps signatures usable as dict keys in the passes.
    """

    dtype: DType
    shape: Shape = Shape.SCALAR

    def __getitem__(self, key: Any) -> Signature:
        keys = key if isinstance(key, tuple) else (key,)
        if len(keys) > 1:
            raise NotImplementedError(f"only 1-d series are supported, got {len(keys)} dimensions")
        if not keys or not isinstance(keys[0], slice):
            raise DslTypeError(f"expected a slice, as in {self.dtype.value}[:], got {key!r}")
        return Signature(self.dtype, Shape.VECTOR)

    @property
    def table(self) -> Signature:
        return Signature(self.dtype, Shape.TABLE)

    def __repr__(self) -> str:
        suffix = {Shape.VECTOR: "[:]", Shape.TABLE: ".table"}.get(self.shape, "")
        return f"{self.dtype.value}{suffix}"


F4 = Signature(DType.FLOAT32)
I4 = Signature(DType.INT32)
B1 = Signature(DType.BOOL)
