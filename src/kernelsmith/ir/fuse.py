"""Elementwise fusion.

Backends emit one loop per elementwise expression, and every loop costs a full
read and write of a series-sized buffer. Merging expressions that can share a
loop turns those intermediates into registers, which on a memory-bound workload
is the whole game.

Fusability is a property of the dependency graph, not of the order the scheduler
happened to pick: two expressions with a feature call between them in the
schedule still belong in one loop as long as neither depends on the call. So
groups are grown backwards from a root, absorbing a producer only when *every*
consumer of its value is already inside the group - which is what keeps the
group convex and lets its internal values become locals.

Feature calls are opaque and never absorbed: the compiler cannot see inside
their loops, and a rolling window is not elementwise anyway.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from kernelsmith.dsl import Expr, Op, ValueNode

# the two DSL operator names that are not what you would write
_DISPLAY_SYMBOL = {"neg": "-", "~": "~"}


class _TooWide(Exception):
    """More distinct inputs than there are letters - stop rendering."""


class FusedExpr(Op):
    """A set of elementwise expressions that share one loop.

    ``members`` is in topological order, so a backend can emit one local
    assignment per member and have every operand already defined. Only values
    consumed from outside appear in ``args``; only the root's result is written
    to a buffer, which is where the saving comes from.
    """

    def __init__(self, root: Expr):
        self.root = root
        self.members: List[Expr] = [root]
        self._member_set = {root}
        self._args: Optional[Tuple[ValueNode, ...]] = None

    # -- construction (pass-internal) --------------------------------------

    def absorb(self, producer: Expr) -> None:
        self.members.append(producer)
        self._member_set.add(producer)
        self._args = None

    def contains(self, op: Op) -> bool:
        return op in self._member_set

    def finalise(self, order: Dict[Op, int]) -> "FusedExpr":
        """Put members in schedule order, so operands precede their uses."""
        self.members.sort(key=lambda member: order[member])
        self._args = None
        return self

    # -- Op interface -------------------------------------------------------

    @property
    def name(self) -> str:
        return f"fused[{len(self.members)}]"

    def formula(self, max_leaves: int = 26) -> Optional[str]:
        """The group as one expression, with leaves lettered a, b, c...

        Returns None past ``max_leaves`` distinct inputs - a formula that wide
        is not worth reading, and the letters run out at z. A value produced
        inside the group appears inline; a value read from a buffer gets a
        letter. Shared internal values are printed at each use, which is what
        you want when reading a formula rather than counting work.
        """
        produced = {member.outs[0]: member for member in self.members}
        letters: Dict[ValueNode, str] = {}

        def walk(member: Expr) -> str:
            parts = []
            for operand in member.args:
                inner = produced.get(operand)
                if inner is not None:
                    parts.append(walk(inner))
                    continue
                if operand not in letters:
                    if len(letters) >= max_leaves:
                        raise _TooWide
                    letters[operand] = chr(97 + len(letters))
                parts.append(letters[operand])

            symbol = _DISPLAY_SYMBOL.get(member.name, member.name)
            if len(parts) == 1:
                return f"({symbol}{parts[0]})"
            return f"({parts[0]} {symbol} {parts[1]})"

        try:
            return walk(self.root)
        except _TooWide:
            return None

    @property
    def produced(self) -> Tuple[ValueNode, ...]:
        """Values computed inside the group - locals, not buffers."""
        return tuple(member.outs[0] for member in self.members)

    @property
    def args(self) -> Tuple[ValueNode, ...]:
        if self._args is None:
            internal = set(self.produced)
            external = []
            for member in self.members:
                for operand in member.args:
                    if operand not in internal:
                        external.append(operand)
            # dict.fromkeys dedupes without the run-to-run reordering a set
            # would introduce - two members often read the same value
            self._args = tuple(dict.fromkeys(external))
        return self._args

    @property
    def outs(self) -> Tuple[ValueNode, ...]:
        return self.root.outs

    def __repr__(self):
        formula = self.formula()
        if formula is None:
            return f"<FusedExpr {len(self.members)} ops>"
        return f"<FusedExpr {len(self.members)} ops: {formula}>"


def consumers(ops: List[Op], replace: Optional[Dict[ValueNode, ValueNode]] = None):
    """value -> the ops that read it, in schedule order."""
    replace = replace or {}
    found: Dict[ValueNode, List[Op]] = {}
    for op in ops:
        for operand in op.args:
            operand = replace.get(operand, operand)
            found.setdefault(operand, []).append(op)
    return found


def fuse(
    ops: List[Op],
    outputs=(),
    replace: Optional[Dict[ValueNode, ValueNode]] = None,
) -> Tuple[List[Op], Dict[Op, int]]:
    """Group elementwise expressions that can share a loop.

    ``outputs`` are the graph's registered output values. They have to be named
    explicitly: a registered output can easily be consumed only by ops inside a
    group, and absorbing it would turn the value the caller asked for into a
    local that never reaches a buffer.

    Returns the rebuilt schedule and its dependency levels. Ops that were not
    fused are passed through unchanged; a group of one collapses back to the
    plain Expr, so nothing downstream has to special-case it.
    """
    replace = replace or {}
    exported = {replace.get(value, value) for value in outputs}
    reads = consumers(ops, replace)
    order = {op: i for i, op in enumerate(ops)}

    grouped: Dict[Op, FusedExpr] = {}
    groups: List[FusedExpr] = []

    # backwards from the roots: a root is an expression nobody has claimed yet
    for op in reversed(ops):
        if not isinstance(op, Expr) or op in grouped:
            continue

        group = FusedExpr(op)
        grouped[op] = group
        pending = [op]

        while pending:
            current = pending.pop()
            for operand in current.args:
                operand = replace.get(operand, operand)
                producer = operand.parent

                if not isinstance(producer, Expr) or producer in grouped:
                    continue
                # a scalar has no loop index to share
                if producer.outs[0].shape is not op.outs[0].shape:
                    continue
                # the caller is going to read this one out of its buffer
                if operand in exported:
                    continue
                # it may only become a local if nobody outside the group still
                # needs it in a buffer
                if any(not group.contains(reader) for reader in reads.get(operand, ())):
                    continue

                group.absorb(producer)
                grouped[producer] = group
                pending.append(producer)

        groups.append(group.finalise(order))

    new_ops: List[Op] = []
    for op in ops:
        group = grouped.get(op)
        if group is None:
            new_ops.append(op)
        elif len(group.members) == 1:
            new_ops.append(op)                       # nothing gained, keep the Expr
        elif op is group.root:
            # safe to emit here: the root consumes its members, so it is last
            # among them in schedule order
            new_ops.append(group)

    levels: Dict[Op, int] = {}
    producer_of: Dict[ValueNode, Op] = {}
    for op in new_ops:
        level = 0
        for operand in op.args:
            operand = replace.get(operand, operand)
            source = producer_of.get(operand)
            if source is not None:
                level = max(level, levels[source] + 1)
        levels[op] = level

        produced = op.produced if isinstance(op, FusedExpr) else op.outs
        for value in produced:
            producer_of[value] = op

    return new_ops, levels
