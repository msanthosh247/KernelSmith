from kernelsmith.dsl import ValueNode , Op
from typing import Dict , List , Tuple


class Liveness:
    def __init__(self , ops : List[Op], outputs : Dict[str , ValueNode]    , replace : Dict[ValueNode , ValueNode] = None):   
        self.ops  = ops
        if replace is None:
            replace = dict()
        self.replace = replace
        self.outputs = {self.replace.get(v, v) for v in outputs.values()} #set(list(outputs.values()))
        
        self.define_index : Dict[ValueNode , int] = dict()
        self.last_use : Dict[ValueNode , int] = dict()
        self.dead : set[ValueNode] = set()
        self.all_nodes : set[ValueNode] = set()
        self.build()

    def live_at(self, i):
        return {
            v for v in self.all_nodes
            if self.define_index[v] <= i <= self.last_use[v]
        }

    def build(self):
        for i ,  op in enumerate(self.ops):
            for arg in op.args:
                arg = self.replace.get(arg , arg)
                self.all_nodes.add(arg)
                if arg not in self.define_index:
                    self.define_index[arg] = -1
                self.last_use[arg] =  i

            for out in op.outs:
                self.all_nodes.add(out)
                self.define_index[out] = i 

        for out in self.outputs:
            self.last_use[out] = len(self.ops)


        for node in self.define_index:
            if (
                not (node  in  self.last_use)
            ):
                self.dead.add(node)
                self.last_use[node] = self.define_index[node]

    


        





            
            
            





    