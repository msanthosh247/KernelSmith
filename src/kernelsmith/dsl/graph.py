"""Graph construction layer: value nodes, operations, and the Graph container.

Value-centric SSA: every ValueNode is created exactly once by its producer,
so output overwrites and cycles are unrepresentable by construction. A node
carries WHAT it is (dtype/shape/role) and WHO made it (parent) - never WHERE
it lives. Buffer locations belong to the allocator pass.

Operations (Expr for elementwise math, Call for feature invocations) share the
Op interface - name, args, outs, buffer_signature - so passes and backends are
written once instead of once per operation kind.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from kernelsmith.errors import DslTypeError, GraphError
from kernelsmith.dsl.types import (
    CONST_OPERAND_TYPES,
    DType,
    Shape,
    ValueSignature,
    VarRole,
    infer_dtype,
    promote_dtype,
    result_shape,
)


class ValueNode:
    """A single value (virtual register) in the graph."""

    def __init__(
        self,
        dtype: DType,
        shape: Shape,
        role: VarRole,
        name: Optional[str] = None,
        val: float | int | bool | None = None,
        parent: "Op | None" = None,
        out_index: int = 0,
    ):
        self.dtype = dtype
        self.shape = shape
        self.role = role
        self.name = name
        self.val = val
        self.parent = parent
        self.out_index = out_index

    def _wrap(self, other) -> "ValueNode":
        if isinstance(other, ValueNode):
            return other
        if isinstance(other, CONST_OPERAND_TYPES):
            return ValueNode(infer_dtype(other), Shape.SCALAR, VarRole.CONST, val=other)
        raise DslTypeError(f"invalid operand of type {type(other).__name__}")

    def _binary_op(self, other, operation: str, self_at_right: bool = False) -> "ValueNode":
        other = self._wrap(other)
        left, right = (other, self) if self_at_right else (self, other)
        return Expr(operation, left, right).output

    def _unary_op(self, operation: str) -> "ValueNode":
        return Expr(operation, self, None).output

    def __gt__(self, other): return self._binary_op(other, ">")
    def __ge__(self, other): return self._binary_op(other, ">=")
    def __lt__(self, other): return self._binary_op(other, "<")
    def __le__(self, other): return self._binary_op(other, "<=")
    def __eq__(self, other): return self._binary_op(other, "==")
    def __add__(self, other): return self._binary_op(other, "+")
    def __radd__(self, other): return self._binary_op(other, "+", self_at_right=True)
    def __sub__(self, other): return self._binary_op(other, "-")
    def __rsub__(self, other): return self._binary_op(other, "-", self_at_right=True)
    def __mul__(self, other): return self._binary_op(other, "*")
    def __rmul__(self, other): return self._binary_op(other, "*", self_at_right=True)
    def __truediv__(self, other): return self._binary_op(other, "/")
    def __rtruediv__(self, other): return self._binary_op(other, "/", self_at_right=True)
    def __and__(self, other): return self._binary_op(other, "&")
    def __or__(self, other): return self._binary_op(other, "|")
    def __xor__(self, other): return self._binary_op(other, "^")
    def __neg__(self): return self._unary_op("neg")
    def __invert__(self): return self._unary_op("~")

    # defining __eq__ would otherwise set __hash__ to None;
    # identity hash keeps nodes usable as dict keys in the passes
    __hash__ = object.__hash__

    def __repr__(self):
        tag = self.name or (repr(self.val) if self.role is VarRole.CONST else "")
        return f"<ValueNode {self.role.value} {self.shape.value} {self.dtype.value} {tag}".rstrip() + ">"


class Op:
    """Base class for graph operations.

    Passes and backends see only this interface:

    ``name``              display name ("+" for an Expr, "sma" for a Call)
    ``args``              values consumed, in signature order
    ``outs``              values produced, in signature order
    ``buffer_signature``  scratch space the op needs, provisioned by the allocator

    dtype and shape deliberately live on the *values*, not on the op: a
    multi-output op has no single dtype.
    """

    name: str
    args: Tuple[ValueNode, ...] = ()
    outs: Tuple[ValueNode, ...] = ()
    buffer_signature: Tuple[ValueSignature, ...] = ()

    def __repr__(self):
        return f"<{type(self).__name__} '{self.name}'>"


class Expr(Op):
    """One application of an elementwise operation. ``right`` is None for unary ops.

    Elementwise ops are *transparent*: the compiler can see the formula and
    inline it, which is what makes them fusable (unlike an opaque Call).
    """

    def __init__(self, operation: str, left: ValueNode, right: Optional[ValueNode]):
        self.name = operation
        self.left = left
        self.right = right
        self.args = (left,) if right is None else (left, right)

        dtype = promote_dtype(
            left.dtype, right.dtype if right is not None else None, operation
        )
        shape = result_shape(left.shape, right.shape if right is not None else None)
        self.outs = (ValueNode(dtype, shape, VarRole.TEMP, parent=self),)

    @property
    def output(self) -> ValueNode:
        """Convenience for the DSL construction path; passes should use ``outs``."""
        return self.outs[0]


class Call(Op):
    """One invocation of a CallFactory. The factory is the immutable spec,
    the Call carries the per-use state (args, outs)."""

    def __init__(self, factory: "CallFactory", args: Tuple[ValueNode, ...]):
        self.factory = factory
        self.name = factory.func_name
        self.args = tuple(args)
        self.outs: Tuple[ValueNode, ...] = ()

    @property
    def buffer_signature(self) -> Tuple[ValueSignature, ...]:
        return tuple(self.factory.buffer_signature)


class CallFactory:
    """Immutable spec of a feature: signatures only.

    Each __call__ makes a fresh Call - never store per-use state on the factory.
    Implementations live in the backends, keyed by factory, so adding a backend
    touches no DSL code.
    """

    def __init__(
        self,
        func_name: str,
        input_signature: List[ValueSignature],
        buffer_signature: List[ValueSignature],
        output_signature: List[ValueSignature],
    ):
        self.func_name = func_name
        self.input_signature = list(input_signature)
        # scratch space the kernel needs; the allocator provisions these, no nodes required
        self.buffer_signature = list(buffer_signature)
        self.output_signature = list(output_signature)

    def __call__(self, *inputs) -> "ValueNode | Tuple[ValueNode, ...]":
        if len(inputs) != len(self.input_signature):
            raise DslTypeError(
                f"'{self.func_name}' expects {len(self.input_signature)} arguments, got {len(inputs)}"
            )

        wrapped = []
        for i, (sig, inp) in enumerate(zip(self.input_signature, inputs)):
            if not isinstance(inp, ValueNode):
                if isinstance(inp, CONST_OPERAND_TYPES):
                    inp = ValueNode(infer_dtype(inp), Shape.SCALAR, VarRole.CONST, val=inp)
                else:
                    raise DslTypeError(
                        f"'{self.func_name}' argument {i}: expected a ValueNode, got {type(inp).__name__}"
                    )
            if sig.dtype is not inp.dtype or sig.shape is not inp.shape:
                raise DslTypeError(
                    f"'{self.func_name}' argument {i}: expected {sig.shape.value} {sig.dtype.value},"
                    f" got {inp.shape.value} {inp.dtype.value}"
                )
            wrapped.append(inp)

        call = Call(self, tuple(wrapped))
        call.outs = tuple(
            ValueNode(osig.dtype, osig.shape, VarRole.TEMP, parent=call, out_index=i)
            for i, osig in enumerate(self.output_signature)
        )
        return call.outs[0] if len(call.outs) == 1 else call.outs

    def __repr__(self):
        return f"<CallFactory '{self.func_name}'>"


class Graph:
    """Container for one computation: named inputs, params and outputs.

    The graph itself is discovered by walking backwards from the registered
    outputs - values reference their producers, producers reference their args.
    """

    def __init__(self):
        self.inputs: dict = {}          # name -> ValueNode
        self.params: dict = {}          # name -> ValueNode
        self.outputs: dict = {}         # name -> ValueNode
        self.output_names: dict = {}    # ValueNode -> name  (inverse of outputs)
        self.ops: List[Op] = []
        self.op_levels: dict = {}

    def input(self, name: str, dtype: DType = DType.FLOAT32) -> ValueNode:
        if name in self.inputs:
            return self.inputs[name]
        node = ValueNode(dtype, Shape.VECTOR, VarRole.INPUT, name=name)
        self.inputs[name] = node
        return node

    def _param(self, name: str, dtype: DType) -> ValueNode:
        if name in self.params:
            node = self.params[name]
            if node.dtype is not dtype:
                raise GraphError(f"param '{name}' already exists with dtype {node.dtype.value}")
            return node
        node = ValueNode(dtype, Shape.SCALAR, VarRole.PARAM, name=name)
        self.params[name] = node
        return node

    def int_param(self, name: str) -> ValueNode:
        return self._param(name, DType.INT32)

    def float_param(self, name: str) -> ValueNode:
        return self._param(name, DType.FLOAT32)

    def output(self, name: str, node: ValueNode):
        if not isinstance(node, ValueNode):
            raise GraphError(
                f"output '{name}' must be a single ValueNode, got {type(node).__name__}"
                " - unpack multi-output results first"
            )
        if name in self.outputs:
            raise GraphError(f"duplicate output name '{name}'")
        self.outputs[name] = node
        self.output_names[node] = name

    def is_output(self, node: ValueNode) -> bool:
        """True when ``node`` is registered as an output (never test the dicts directly)."""
        return node in self.output_names

    def build(self) -> List[Op]:
        """Discover every op reachable from the outputs, in topological order.

        Walks backwards (ValueNode.parent is the producing op, op.args are the
        consumed values) with an iterative post-order DFS, so producers always
        come before consumers. No cycle check needed: SSA construction makes
        cycles unrepresentable.
        """
        if not self.outputs:
            raise GraphError("no outputs registered - call graph.output(name, node) first")

        self.ops = []
        visited = set()
        stack = []

        for node in self.outputs.values():
            if node.parent is not None:
                stack.append((node.parent, False))

        while stack:
            op, children_done = stack.pop()
            if children_done:
                self.ops.append(op)
                continue
            if op in visited:
                continue
            visited.add(op)
            stack.append((op, True))
            for arg in op.args:
                if (arg.parent is not None) and (arg.parent not in visited):
                    stack.append((arg.parent, False))

        # dependency level of every op, roots at 0 (feeds the layered plot, later the scheduler)
        self.op_levels = {}
        for op in self.ops:
            lvl = 0
            for arg in op.args:
                if arg.parent is not None:
                    lvl = max(lvl, self.op_levels[arg.parent] + 1)
            self.op_levels[op] = lvl

        return self.ops

    def visualize(self, savepath: Optional[str] = None, figsize=None):
        """Layered plot of the call graph.

        Circles = ValueNodes (colored by role), squares = feature Calls,
        diamonds = Exprs, red outline = registered outputs. Requires the
        optional viz dependencies (networkx, matplotlib).
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        if not self.ops:
            self.build()

        _dt = {DType.FLOAT32: "f32", DType.INT32: "i32", DType.BOOL: "b1"}

        def _value_label(v: ValueNode) -> str:
            if self.is_output(v):
                return f"{self.output_names[v]}\n{_dt[v.dtype]}"
            if v.role in (VarRole.INPUT, VarRole.PARAM):
                return f"{v.name}\n{_dt[v.dtype]}"
            if v.role is VarRole.CONST:
                return f"{v.val}\n{_dt[v.dtype]}"
            return f"{_dt[v.dtype]}\n{v.shape.value[:3]}"

        G = nx.DiGraph()
        labels = {}

        def _add_value(v: ValueNode):
            if id(v) not in G:
                layer = 0 if v.parent is None else 2 * self.op_levels[v.parent] + 2
                G.add_node(id(v), obj=v, kind="value", layer=layer)
                labels[id(v)] = _value_label(v)

        for op in self.ops:
            # the one place the operation kind genuinely matters: node shape
            G.add_node(
                id(op),
                obj=op,
                kind="call" if isinstance(op, Call) else "expr",
                layer=2 * self.op_levels[op] + 1,
            )
            labels[id(op)] = op.name

            for arg in op.args:
                _add_value(arg)
                G.add_edge(id(arg), id(op))

            for out in op.outs:
                _add_value(out)
                G.add_edge(id(op), id(out))

        pos = nx.multipartite_layout(G, subset_key="layer")

        if figsize is None:
            layers = [d["layer"] for _, d in G.nodes(data=True)]
            colmax = max(sum(1 for l in layers if l == c) for c in set(layers))
            figsize = (max(9, 1.8 * len(set(layers))), max(5, 1.3 * colmax))
        plt.figure(figsize=figsize)

        role_colors = {
            VarRole.INPUT: "#9ed49e",
            VarRole.PARAM: "#9ec2e8",
            VarRole.CONST: "#d9d9d9",
            VarRole.TEMP: "#f5f0c8",
        }

        value_ids = [n for n, d in G.nodes(data=True) if d["kind"] == "value"]
        for role, color in role_colors.items():
            ids = [n for n in value_ids if G.nodes[n]["obj"].role is role]
            if not ids:
                continue
            nx.draw_networkx_nodes(
                G, pos, nodelist=ids, node_shape="o", node_color=color, node_size=1500,
                edgecolors=["#c0392b" if self.is_output(G.nodes[n]["obj"]) else "#444444" for n in ids],
                linewidths=[2.5 if self.is_output(G.nodes[n]["obj"]) else 1.0 for n in ids],
            )

        expr_ids = [n for n, d in G.nodes(data=True) if d["kind"] == "expr"]
        if expr_ids:
            nx.draw_networkx_nodes(
                G, pos, nodelist=expr_ids, node_shape="D", node_color="#cdb6e8",
                node_size=1200, edgecolors="#444444", linewidths=1.0,
            )

        call_ids = [n for n, d in G.nodes(data=True) if d["kind"] == "call"]
        if call_ids:
            nx.draw_networkx_nodes(
                G, pos, nodelist=call_ids, node_shape="s", node_color="#f2b06b",
                node_size=2000, edgecolors="#444444", linewidths=1.0,
            )

        nx.draw_networkx_edges(G, pos, edge_color="#888888", arrows=True, arrowsize=14, node_size=1500)
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        legend = [
            Line2D([], [], marker="o", color="w", markerfacecolor=c, markeredgecolor="#444444",
                   markersize=11, label=f"value : {r.value}")
            for r, c in role_colors.items()
        ]
        legend += [
            Line2D([], [], marker="D", color="w", markerfacecolor="#cdb6e8",
                   markeredgecolor="#444444", markersize=10, label="Expr"),
            Line2D([], [], marker="s", color="w", markerfacecolor="#f2b06b",
                   markeredgecolor="#444444", markersize=11, label="Call (feature)"),
            Line2D([], [], marker="o", color="w", markerfacecolor="w",
                   markeredgecolor="#c0392b", markersize=11, label="registered output"),
        ]
        plt.legend(handles=legend, loc="upper left", fontsize=8, frameon=False)
        plt.axis("off")
        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
            plt.close()
            return savepath
        plt.show()
