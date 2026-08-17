"""Human-readable dump of a compiled graph.

Shows what every pass decided in one table: the schedule, each op's kind and
dependency level, where its results live, the scratch it borrows, and which
values are produced but never read.
"""
from typing import List

from kernelsmith.dsl import Call, DType, Graph, Op, Shape, Signature, ValueNode
from kernelsmith.ir.allocate import Allocation, Slot, allocate
from kernelsmith.ir.cse import cse
from kernelsmith.ir.fuse import FusedExpr, fuse
from kernelsmith.ir.liveness import Liveness

_DTYPE_TAG = {DType.FLOAT32: "f32", DType.INT32: "i32", DType.BOOL: "b1"}
_SHAPE_TAG = {Shape.VECTOR: "v", Shape.SCALAR: "s"}



def _slot_index(slot: Slot) -> str:
    return "-" if slot is None else str(slot[1])


def _value_tag(value: ValueNode, allocation: Allocation, live: Liveness) -> str:
    """O:f32v[2] - an output, float32 vector, buffer 2 of the output pool."""
    kind = "O" if value in live.outputs else "T"
    slot = allocation.slots.get(value)
    tag = f"{kind}:{_DTYPE_TAG[value.dtype]}{_SHAPE_TAG[value.shape]}[{_slot_index(slot)}]"
    return tag + "*" if value in live.dead else tag


def _scratch_tag(signature: Signature, slot: Slot) -> str:
    return f"{_DTYPE_TAG[signature.dtype]}{_SHAPE_TAG[signature.shape]}[{_slot_index(slot)}]"




def explain(
    graph: Graph,
    ops: List[Op],
    live: Liveness,
    allocation: Allocation,
    levels: dict = None,
) -> str:
    levels = graph.op_levels if levels is None else levels
    def _row(*cells) -> str:
        return "".join(str(cell).ljust(width) for cell, width in zip(cells, _WIDTHS)).rstrip()

    _WIDTHS = (4, 5, 6, 18, 30, 14)
    lines = [
        f"{len(ops)} ops   "
        f"{len(graph.inputs)} input(s)   "
        f"{len(graph.params)} param(s)   "
        f"{len(graph.outputs)} output(s)",
        "",
        _row("op", "lvl", "kind", "name", "outs", "scratch"),
        _row(*("-" * (width - 1) for width in _WIDTHS)),
    ]

    for i, op in enumerate(ops):
        outs = " ".join(_value_tag(out, allocation, live) for out in op.outs)
        scratch = " ".join(
            _scratch_tag(signature, slot)
            for signature, slot in zip(op.buffer_signature, allocation.scratch.get(op, ()))
        )
        if isinstance(op, Call):
            kind = "call"
        elif isinstance(op, FusedExpr):
            kind = "fused"
        else:
            kind = "expr"
        lines.append(
            _row(i, levels.get(op, "?"), kind, op.name, outs or "-", scratch or "-")
        )

    lines += ["", "pools:"]
    for key in sorted(allocation.pool_size, key=lambda k: (k[0].value, k[1].value, k[2].value)):
        kind, dtype, shape = key
        lines.append(f"  {kind.value}/{dtype.value}/{shape.value}  x{allocation.pool_size[key]}")

    if live.dead:
        lines += ["", "* produced but never read - scratch space, never copied back"]

    return "\n".join(lines)


def explain_graph(graph: Graph, fused: bool = True) -> str:
    """Run the whole pipeline and describe the result."""
    ops, replace = cse(graph.build())
    levels = graph.op_levels
    if fused:
        ops, levels = fuse(ops, graph.outputs.values(), replace)
    live = Liveness(ops, graph.outputs, replace)
    return explain(graph, ops, live, allocate(ops, live), levels)
