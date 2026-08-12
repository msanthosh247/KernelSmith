"""Linear-scan buffer allocation.

The graph uses one virtual register per value; hardware wants a small set of
reused buffers. This pass walks the schedule once and hands every value a slot
in a pool, recycling a slot as soon as the value's live interval ends.

Pools are keyed by (kind, dtype, shape). Temp pools are recycled; the output
pool is not, because registered outputs must survive to be copied back. Scratch
declared by a feature's buffer_signature is taken and returned within one op,
so it recycles through the temp pool like anything else.

Inputs, params and constants are never allocated - the caller provides them.
"""
from kernelsmith.dsl import ValueNode , Graph , ValueSignature , DType , Op , Shape , VarRole
from kernelsmith.ir import Liveness
from kernelsmith.errors import KernelsmithError
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List , Tuple

class PoolKind(Enum):
    TEMP = "temp"
    OUTPUT = "output"

PoolKey = Tuple[PoolKind , DType , Shape]
Slot = Tuple[PoolKey , int]


@dataclass
class Allocation:
    slots : Dict[ValueNode  , Slot]
    scratch : Dict[Op , Tuple[Slot , ...]]
    pool_size : Dict[PoolKey , int]

class PoolTracker:
    def __init__(self):
        self._free : Dict[PoolKey , List[int]] = dict()
        self._next : Dict[PoolKey , int] = dict()

    def take(self, key : PoolKey) -> Slot:
        if not key in self._free:
            self._free[key] = []
        if len(self._free[key]) == 0 :
            self._next[key] = self._next.get(key , 0) + 1
            return (key , self._next[key] - 1)
        else:
            nfree = self._free[key].pop()
            return (key , nfree)

    def release(self, slot : Slot) -> None:
        key , index = slot
        free = self._free.setdefault(key , [])
        # a slot handed out twice would silently give two live values the same
        # buffer , which nothing downstream can detect
        if index in free:
            raise KernelsmithError(f"slot {key}[{index}] released twice")
        free.append(index)

    def sizes(self) -> Dict[PoolKey , int]:
        return dict(self._next)


def _allocatable(value : ValueNode) -> bool:
    """Only produced values need a buffer; inputs , params and consts are given."""
    return value.role is VarRole.TEMP


def _key(value : ValueNode , live : Liveness) -> PoolKey:
    kind = PoolKind.OUTPUT if value in live.outputs else PoolKind.TEMP
    return (kind , value.dtype , value.shape)


def _scratch_key(signature : ValueSignature) -> PoolKey:
    return (PoolKind.TEMP , signature.dtype , signature.shape)


def allocate(ops : List[Op] , live : Liveness) -> Allocation:
    """Map every produced value onto a pool slot.

    Within one op the order is fixed : allocate the outputs , take and return
    the scratch , then release whatever died here. Releasing last is what stops
    an op from being handed its own argument's buffer to write into while it is
    still reading from it.
    """
    pools = PoolTracker()
    slots : Dict[ValueNode , Slot] = dict()
    scratch : Dict[Op , Tuple[Slot , ...]] = dict()

    for i , op in enumerate(ops):
        # 1. outputs first - they are written while the args are still live
        for out in op.outs:
            if _allocatable(out):
                slots[out] = pools.take(_key(out , live))

        # 2. scratch for this op , and 3. it dies with the op
        scratch[op] = tuple(pools.take(_scratch_key(sig)) for sig in op.buffer_signature)
        for slot in scratch[op]:
            pools.release(slot)

        # 4. release whatever ends here. args AND outs : a dead value is an
        # output whose last use is its own definition , so it never appears as
        # anyone's argument. args go through the CSE map first - a kept op can
        # still name a value whose producer was dropped. dict.fromkeys dedupes
        # 'x * x' , where releasing twice would put one index in the free list
        # twice
        resolved_args = tuple(live.replace.get(arg , arg) for arg in op.args)
        for value in dict.fromkeys(resolved_args + op.outs):
            if not _allocatable(value):
                continue
            if value in live.outputs:
                continue
            if live.last_use[value] == i:
                pools.release(slots[value])

    return Allocation(slots = slots , scratch = scratch , pool_size = pools.sizes())
