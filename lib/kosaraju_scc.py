from typing import List, Set
from .digraph import Digraph
from .depth_first_order import DepthFirstOrder


class KosarajuSCC:
    def __init__(self, G: Digraph) -> None:
        self.marked = [False for _ in range(G.V)]
        self.id = [0 for _ in range(G.V)]
        self.order = DepthFirstOrder(G.reverse())
        self.count = 0
        for s in self.order.get_reverse_post():
            if not self.marked[s]:
                self.dfs(G, s)
                self.count += 1

    def dfs(self, G: Digraph, v: int) -> None:
        self.marked[v] = True
        self.id[v] = self.count

        for w in G.get_adj(v):
            if not self.marked[w]:
                self.dfs(G, w)

    def get_scc(self) -> List[Set[int]]:
        scc = [set() for _ in range(self.count)]
        for vertex, component in enumerate(self.id):
            scc[component].add(vertex)

        return scc