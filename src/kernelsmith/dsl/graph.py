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
    Signature,
    VarRole,
    result_dtype,
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

    @property
    def signature(self) -> Signature:
        """Derived, not stored: it cannot drift from dtype/shape, and graphs
        with thousands of values do not pay for an extra object each."""
        return Signature(self.dtype, self.shape)

    def _wrap(self, other) -> "ValueNode":
        if isinstance(other, ValueNode):
            return other
        if isinstance(other, CONST_OPERAND_TYPES):
            return ValueNode(DType.infer_from_constant(other), Shape.SCALAR, VarRole.CONST, val=other)
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
    def __ne__(self , other): return self._binary_op(other , "!=")
    def __neg__(self): return self._unary_op("neg")
    def __invert__(self): return self._unary_op("~")

    def __getitem__(self, index) -> "ValueNode":
        """``series[i]``: element ``i`` of this series, one value per parameter set.

        ``i`` is an int or an int param; negative counts from the end, and an
        index past either end gives NaN. It is how a simulated path is read at
        an expiry: ``path[20]`` is the value after 21 steps.
        """
        if self.shape is not Shape.VECTOR:
            raise DslTypeError(
                f"only a series can be indexed; {self._describe()} is a {self.shape.value}"
            )
        if self.dtype is not DType.FLOAT32:
            raise DslTypeError(f"indexing supports float32 series, got {self.dtype.value}")
        return element(self, index)

    def _describe(self) -> str:
        return f"'{self.name}'" if self.name else "this value"

    # defining __eq__ would otherwise set __hash__ to None;
    # identity hash keeps nodes usable as dict keys in the passes
    __hash__ = object.__hash__

    def __repr__(self):
        tag = self.name or (repr(self.val) if self.role is VarRole.CONST else "")
        return f"<ValueNode {self.role.value} {self.shape.value} {self.dtype.value} {tag}".rstrip() + ">"

    def __bool__(self):
        raise DslTypeError(
            """
            the truth value of a ValueNode is ambiguous - it is a graph node , not a
            concrete value. Use 'is' for identity, or a set / dict for membership
            """
        )



        

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
    buffer_signature: Tuple[Signature, ...] = ()

    def __repr__(self):
        return f"<{type(self).__name__} '{self.name}'>"


class Expr(Op):
    """One application of an elementwise operation. ``right`` is None for unary ops.

    Elementwise ops are *transparent*: the compiler can see the formula and
    inline it, which is what makes them fusable (unlike an opaque Call).
    """

    def __init__(self, operation: str, left: ValueNode, right: Optional[ValueNode]):
        for operand in (left, right):
            if operand is not None and operand.shape is Shape.TABLE:
                raise DslTypeError(
                    f"{operand._describe()} is a table (reference data): it can only be passed"
                    " to features such as bootstrap_path. To compute on it bar by bar,"
                    " register it with register_input."
                )
        self.name = operation
        self.left = left
        self.right = right
        self.args = (left,) if right is None else (left, right)

        dtype = result_dtype(
            left.dtype, right.dtype if right is not None else None, operation
        )
        shape = left.shape.combine(right.shape if right is not None else None)
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
    def buffer_signature(self) -> Tuple[Signature, ...]:
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
        input_signature: List[Signature],
        buffer_signature: List[Signature],
        output_signature: List[Signature],
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
                    inp = ValueNode(DType.infer_from_constant(inp), Shape.SCALAR, VarRole.CONST, val=inp)
                else:
                    raise DslTypeError(
                        f"'{self.func_name}' argument {i}: expected a ValueNode, got {type(inp).__name__}"
                    )
            if sig.shape is Shape.TABLE and inp.shape is Shape.VECTOR:
                raise DslTypeError(
                    f"'{self.func_name}' argument {i} expects a table ({sig}), got the series"
                    f" {inp._describe()} ({inp.signature}). Register it with register_table:"
                    " a series sits on the graph's time axis, so every output would be as"
                    " long as it."
                )
            if sig != inp.signature:
                raise DslTypeError(
                    f"'{self.func_name}' argument {i}: expected {sig},"
                    f" got {inp.signature}"
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


# The one feature the DSL itself uses: ``series[i]``. Its kernels live with the
# rest in ``kernelsmith.features``, which every backend loads when compiling.
element = CallFactory(
    "element",
    input_signature=[Signature(DType.FLOAT32, Shape.VECTOR), Signature(DType.INT32)],
    buffer_signature=[],
    output_signature=[Signature(DType.FLOAT32)],
)


class Graph:
    """Container for one computation: named inputs, params and outputs.

    The graph itself is discovered by walking backwards from the registered
    outputs - values reference their producers, producers reference their args.
    """

    def __init__(self):
        self.inputs: dict = {}          # name -> ValueNode, series on the time axis
        self.tables: dict = {}          # name -> ValueNode, reference data off it
        self.params: dict = {}          # name -> ValueNode
        self.outputs: dict = {}         # name -> ValueNode
        self.output_names: dict = {}    # ValueNode -> name  (inverse of outputs)
        self.ops: List[Op] = []
        self.op_levels: dict = {}

    def register_input(self, name: str, dtype: DType = DType.FLOAT32) -> ValueNode:
        """A series on the graph's time axis: every bar of it is computed on, and
        its length is the length of every series in the graph."""
        if name in self.inputs:
            return self.inputs[name]
        if name in self.tables:
            raise GraphError(f"'{name}' is already registered as a table")
        node = ValueNode(dtype, Shape.VECTOR, VarRole.INPUT, name=name)
        self.inputs[name] = node
        return node

    def register_table(self, name: str, dtype: DType = DType.FLOAT32) -> ValueNode:
        """Reference data off the time axis: any length, shared by every parameter
        set, and read only by features - e.g. a price history that a simulation
        resamples, while the simulated path is the graph's series.

        Passed to ``run`` in the same dict as the inputs. The name becomes an
        identifier in the generated kernel, so it must be a valid one.
        """
        if name in self.tables:
            return self.tables[name]
        if name in self.inputs:
            raise GraphError(f"'{name}' is already registered as an input")
        if not name.isidentifier():
            raise GraphError(f"table name must be a valid identifier, got '{name}'")
        node = ValueNode(dtype, Shape.TABLE, VarRole.INPUT, name=name)
        self.tables[name] = node
        return node

    def register_param(self, name: str, dtype: DType) -> ValueNode:
        if name in self.params:
            node = self.params[name]
            if node.dtype is not dtype:
                raise GraphError(f"param '{name}' already exists with dtype {node.dtype.value}")
            return node
        node = ValueNode(dtype, Shape.SCALAR, VarRole.PARAM, name=name)
        self.params[name] = node
        return node

    def int_param(self, name: str) -> ValueNode:
        return self.register_param(name, DType.INT32)

    def float_param(self, name: str) -> ValueNode:
        return self.register_param(name, DType.FLOAT32)

    def register_output(self, name: str, node: ValueNode):
        if not isinstance(node, ValueNode):
            raise GraphError(
                f"output '{name}' must be a single ValueNode, got {type(node).__name__}"
                " - unpack multi-output results first"
            )
        if name in self.outputs:
            raise GraphError(f"duplicate output name '{name}'")
        if node.shape is Shape.TABLE:
            raise GraphError(f"output '{name}': a table is reference data you pass in, not a result")
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
            raise GraphError("no outputs registered - call graph.register_output(name, node) first")

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

    def visualize(
        self,
        savepath: Optional[str] = None,
        figsize=None,
        ops: Optional[List[Op]] = None,
        op_levels: Optional[dict] = None,
        replace: Optional[dict] = None,
    ):
        """Layered plot of the call graph.

        Circles = ValueNodes (colored by role, tables in their own color),
        squares = feature Calls (``series[i]`` labelled with its index),
        diamonds = Exprs, hexagons = fused expression groups, red outline =
        registered outputs. Inputs read ``f32[:]`` and tables ``f32 table``,
        so the two differ in the labels as well as the colors. Requires the optional viz dependencies (networkx,
        matplotlib).

        Pass ``ops`` and ``op_levels`` from a pass to plot a transformed
        schedule - fusing first and plotting both is the clearest picture of
        what fusion did, since values that became locals stop being nodes.
        Pass ``replace`` too when the schedule has been through CSE, or
        arguments naming a dropped duplicate will float free of the graph.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        from kernelsmith.ir.fuse import FusedExpr
        # deferred: ir imports dsl, so this cannot be a module-level import

        if not self.ops:
            self.build()
        if ops is None:
            ops, op_levels = self.ops, self.op_levels
        elif op_levels is None:
            raise GraphError("pass op_levels alongside ops - the layout is laid out by level")
        replace = replace or {}

        _dt = {DType.FLOAT32: "f32", DType.INT32: "i32", DType.BOOL: "b1"}

        def _value_label(v: ValueNode) -> str:
            if self.is_output(v):
                return f"{self.output_names[v]}\n{_dt[v.dtype]}"
            if v.shape is Shape.TABLE:
                return f"{v.name}\n{_dt[v.dtype]} table"
            if v.role is VarRole.INPUT:
                return f"{v.name}\n{_dt[v.dtype]}[:]"
            if v.role is VarRole.PARAM:
                return f"{v.name}\n{_dt[v.dtype]}"
            if v.role is VarRole.CONST:
                return f"{v.val}\n{_dt[v.dtype]}"
            return f"{_dt[v.dtype]}\n{v.shape.value[:3]}"

        G = nx.DiGraph()
        labels = {}

        # a fused group produces its members' values as locals, so those values
        # belong to the group rather than to the member that wrote them
        produced_by = {}
        for op in ops:
            for value in (op.produced if isinstance(op, FusedExpr) else op.outs):
                produced_by[value] = op

        def _add_value(v: ValueNode):
            if id(v) not in G:
                source = produced_by.get(v)
                layer = 0 if source is None else 2 * op_levels[source] + 2
                G.add_node(id(v), obj=v, kind="value", layer=layer)
                labels[id(v)] = _value_label(v)

        for op in ops:
            # the one place the operation kind genuinely matters: node shape
            if isinstance(op, Call):
                kind = "call"
            elif isinstance(op, FusedExpr):
                kind = "fused"
            else:
                kind = "expr"
            G.add_node(id(op), obj=op, kind=kind, layer=2 * op_levels[op] + 1)
            label = op.name
            if isinstance(op, Call) and op.factory is element:
                index = replace.get(op.args[1], op.args[1])
                label = f"[{index.val if index.role is VarRole.CONST else index.name}]"
            if isinstance(op, FusedExpr):
                formula = op.formula()
                if formula is not None:
                    label = f"{op.name}\n{formula}"
            labels[id(op)] = label

            for arg in op.args:
                arg = replace.get(arg, arg)
                if arg.role is VarRole.CONST and id(arg) not in G:
                    # every literal is its own node with one consumer: draw it
                    # beside that op, not in the first column, where its edge
                    # would cross the whole plot and appear to come from
                    # whatever it passes behind
                    G.add_node(id(arg), obj=arg, kind="value", layer=2 * op_levels[op])
                    labels[id(arg)] = _value_label(arg)
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

        # a table is an input by role, but drawn apart: it is reference data,
        # not a series the graph computes along
        value_colors = {
            "input": "#9ed49e",
            "table": "#e3b5d6",
            "param": "#9ec2e8",
            "const": "#d9d9d9",
            "temp": "#f5f0c8",
        }

        def _category(v: ValueNode) -> str:
            return "table" if v.shape is Shape.TABLE else v.role.value

        value_ids = [n for n, d in G.nodes(data=True) if d["kind"] == "value"]
        for category, color in value_colors.items():
            ids = [n for n in value_ids if _category(G.nodes[n]["obj"]) == category]
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

        fused_ids = [n for n, d in G.nodes(data=True) if d["kind"] == "fused"]
        if fused_ids:
            nx.draw_networkx_nodes(
                G, pos, nodelist=fused_ids, node_shape="h", node_color="#8fd4d0",
                node_size=2200, edgecolors="#444444", linewidths=1.0,
            )

        nx.draw_networkx_edges(G, pos, edge_color="#888888", arrows=True, arrowsize=14, node_size=1500)
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        legend = [
            Line2D([], [], marker="o", color="w", markerfacecolor=c, markeredgecolor="#444444",
                   markersize=11, label=f"value : {category}")
            for category, c in value_colors.items()
        ]
        legend += [
            Line2D([], [], marker="D", color="w", markerfacecolor="#cdb6e8",
                   markeredgecolor="#444444", markersize=10, label="Expr"),
            Line2D([], [], marker="s", color="w", markerfacecolor="#f2b06b",
                   markeredgecolor="#444444", markersize=11, label="Call (feature)"),
            Line2D([], [], marker="h", color="w", markerfacecolor="#8fd4d0",
                   markeredgecolor="#444444", markersize=12, label="Fused group"),
            Line2D([], [], marker="o", color="w", markerfacecolor="w",
                   markeredgecolor="#c0392b", markersize=11, label="registered output"),
        ]
        # outside the axes, so it can never sit on top of a node
        plt.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.0, 1.0),
                   fontsize=8, frameon=False)
        plt.axis("off")
        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
            plt.close()
            return savepath
        plt.show()
