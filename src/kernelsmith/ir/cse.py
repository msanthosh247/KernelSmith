from kernelsmith.dsl import Op , ValueNode  , Call , VarRole, Expr
from typing import List , Tuple , Dict

def resolve(v , replace):
    return replace.get(v , v)

def head(op : Op):
    return ("call", op.factory) if isinstance(op , Call) else ("expr" , op.name)

def arg_key(v , number_of):
    if v.role is VarRole.CONST:
        return (1 , v.dtype.value , repr(v.val))  #, number_of

    if v not in number_of:
        number_of[v] = len(number_of)
    vnum = number_of[v] 
    return (0 , vnum) #, number_of

COMMUTATIVE = {"+" , "*" , "&" , "|" , "^" , "=="}

def cse(ops : List[Op]) -> Tuple[List[Op] , Dict[ValueNode , ValueNode]]:
    """
    Returns (kept_ops , replace). replace maps each value of a dropped
    duplicate to the corresponding value of the op that was kept.

    This Approach still doesn't support Associativity checks
    """
    canon = {} # fingerprint -> the op we kept 
    replace = {} # dropped value -> canonical value
    kept = []  
    number = {} # ValueNode -> stable int , used only for ordering

    for op in ops: # from graph.ops after topo sort
        keys = [] 
        for a in op.args:
            nk = arg_key(resolve(a , replace) , number) 
            keys.append(nk)
        if isinstance(op , Expr) and op.name in COMMUTATIVE and len(keys) == 2:
            keys = sorted(keys)
        fp = (head(op) , tuple(keys))
        if fp in canon:
            twin = canon[fp]
            for mine , theirs in zip(op.outs , twin.outs):
                replace[mine] = theirs
        else:
            canon[fp] = op
            kept.append(op)
    return kept , replace

        

