import numpy as np
import pytest

from kernelsmith import Graph, Shape
from kernelsmith.dsl.types import (
    B1,
    CONST_OPERAND_TYPES,
    DType,
    F4,
    I4,
    OpCategory,
    Signature,
    result_dtype,
)
from kernelsmith.errors import DslTypeError


# ---- result_dtype ----------------------------------------------------------

def test_arith_promotes_up_the_lattice():
    assert result_dtype(DType.INT32, DType.FLOAT32, "+") is DType.FLOAT32
    assert result_dtype(DType.INT32, DType.INT32, "*") is DType.INT32


def test_true_division_always_float():
    assert result_dtype(DType.INT32, DType.INT32, "/") is DType.FLOAT32


def test_compare_yields_bool():
    assert result_dtype(DType.FLOAT32, DType.INT32, ">") is DType.BOOL


def test_logic_requires_bool():
    assert result_dtype(DType.BOOL, DType.BOOL, "&") is DType.BOOL
    with pytest.raises(DslTypeError, match="did you mean a comparison"):
        result_dtype(DType.FLOAT32, DType.FLOAT32, "&")


def test_arith_rejects_bool():
    with pytest.raises(DslTypeError):
        result_dtype(DType.BOOL, DType.INT32, "+")


def test_unary_ops():
    assert result_dtype(DType.FLOAT32, None, "neg") is DType.FLOAT32
    assert result_dtype(DType.BOOL, None, "~") is DType.BOOL
    with pytest.raises(DslTypeError):
        result_dtype(DType.FLOAT32, None, "~")
    with pytest.raises(DslTypeError):
        result_dtype(DType.BOOL, None, "neg")


def test_operand_count_is_checked():
    with pytest.raises(DslTypeError, match="takes only one operand"):
        result_dtype(DType.BOOL, DType.BOOL, "~")
    with pytest.raises(DslTypeError, match="requires two operands"):
        result_dtype(DType.INT32, None, "+")


def test_unknown_operation():
    with pytest.raises(DslTypeError, match="unknown operation"):
        result_dtype(DType.INT32, DType.INT32, "%")


def test_every_operator_has_a_typing_rule():
    """Adding an OpCategory without a branch in result_dtype must fail loudly,
    not quietly hand back a dtype of None."""
    dtypes = list(DType)
    for category in OpCategory:
        for operation in category.value:
            unary = category is OpCategory.UNARY
            for a in dtypes:
                for b in ([None] if unary else dtypes):
                    try:
                        assert result_dtype(a, b, operation) in DType
                    except DslTypeError:
                        pass          # a type error is a rule; None would not be


# ---- DType -----------------------------------------------------------------

def test_dtype_has_exactly_three_members():
    """Plain attributes in an Enum body become members - this guards against a
    constant being moved into DType by accident."""
    assert set(DType) == {DType.BOOL, DType.INT32, DType.FLOAT32}


def test_join_is_the_wider_dtype():
    assert DType.INT32.join(DType.FLOAT32) is DType.FLOAT32
    assert DType.FLOAT32.join(DType.BOOL) is DType.FLOAT32
    assert DType.BOOL.join(DType.BOOL) is DType.BOOL


def test_is_numeric():
    assert DType.INT32.is_numeric and DType.FLOAT32.is_numeric
    assert not DType.BOOL.is_numeric


def test_infer_from_constant_bool_before_int():
    # python bool subclasses int - the bool check must win
    assert DType.infer_from_constant(True) is DType.BOOL
    assert DType.infer_from_constant(5) is DType.INT32
    assert DType.infer_from_constant(2.5) is DType.FLOAT32
    assert DType.infer_from_constant(np.float32(1.0)) is DType.FLOAT32
    with pytest.raises(DslTypeError):
        DType.infer_from_constant("nope")


def test_const_operand_types_cover_python_and_numpy_scalars():
    for value in (True, 3, 2.5, np.bool_(True), np.int32(3), np.float32(2.5)):
        assert isinstance(value, CONST_OPERAND_TYPES)
    assert not isinstance("x", CONST_OPERAND_TYPES)


# ---- Shape -----------------------------------------------------------------

def test_binary_shapes_broadcast():
    assert Shape.VECTOR.combine(Shape.SCALAR) is Shape.VECTOR
    assert Shape.SCALAR.combine(Shape.VECTOR) is Shape.VECTOR
    assert Shape.SCALAR.combine(Shape.SCALAR) is Shape.SCALAR


def test_unary_keeps_the_operand_shape():
    assert Shape.VECTOR.combine() is Shape.VECTOR
    assert Shape.SCALAR.combine() is Shape.SCALAR


def test_unary_ops_on_a_series_stay_series():
    """End to end: negating or inverting a series must not produce a scalar."""
    g = Graph()
    close = g.register_input("close")
    assert (-close).shape is Shape.VECTOR
    assert (~(close > 100.0)).shape is Shape.VECTOR


# ---- Signature -------------------------------------------------------------

def test_signature_constants_are_module_level_scalars():
    assert F4 == Signature(DType.FLOAT32)
    assert I4.shape is Shape.SCALAR and B1.dtype is DType.BOOL


def test_slicing_gives_the_series_form():
    assert F4[:] == Signature(DType.FLOAT32, Shape.VECTOR)
    assert repr(F4[:]) == "float32[:]"
    assert repr(I4) == "int32"
