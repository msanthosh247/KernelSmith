import numpy as np
import pytest

from kernelsmith.dsl.types import DType, Shape, infer_dtype, promote_dtype, result_shape
from kernelsmith.errors import DslTypeError


def test_arith_promotes_up_the_lattice():
    assert promote_dtype(DType.INT32, DType.FLOAT32, "+") is DType.FLOAT32
    assert promote_dtype(DType.INT32, DType.INT32, "*") is DType.INT32


def test_true_division_always_float():
    assert promote_dtype(DType.INT32, DType.INT32, "/") is DType.FLOAT32


def test_compare_yields_bool():
    assert promote_dtype(DType.FLOAT32, DType.INT32, ">") is DType.BOOL


def test_logic_requires_bool():
    assert promote_dtype(DType.BOOL, DType.BOOL, "&") is DType.BOOL
    with pytest.raises(DslTypeError, match="did you mean a comparison"):
        promote_dtype(DType.FLOAT32, DType.FLOAT32, "&")


def test_arith_rejects_bool():
    with pytest.raises(DslTypeError):
        promote_dtype(DType.BOOL, DType.INT32, "+")


def test_unary_ops():
    assert promote_dtype(DType.FLOAT32, None, "neg") is DType.FLOAT32
    assert promote_dtype(DType.BOOL, None, "~") is DType.BOOL
    with pytest.raises(DslTypeError):
        promote_dtype(DType.FLOAT32, None, "~")
    with pytest.raises(DslTypeError):
        promote_dtype(DType.BOOL, None, "neg")


def test_unknown_operation():
    with pytest.raises(DslTypeError, match="unknown operation"):
        promote_dtype(DType.INT32, DType.INT32, "%")


def test_infer_dtype_bool_before_int():
    # python bool subclasses int - the bool check must win
    assert infer_dtype(True) is DType.BOOL
    assert infer_dtype(5) is DType.INT32
    assert infer_dtype(2.5) is DType.FLOAT32
    assert infer_dtype(np.float32(1.0)) is DType.FLOAT32
    with pytest.raises(DslTypeError):
        infer_dtype("nope")


def test_result_shape_broadcasts():
    assert result_shape(Shape.VECTOR, Shape.SCALAR) is Shape.VECTOR
    assert result_shape(Shape.SCALAR, Shape.SCALAR) is Shape.SCALAR
    assert result_shape(Shape.VECTOR) is Shape.VECTOR
