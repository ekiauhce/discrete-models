from __future__ import annotations
from typing import List
from lib.edge import Edge

class EdgeWeightedGraph:
    def __init__(self, V: int) -> None:
        self.V = V
        self.adj = [ [] for _ in range(V)]
        self.E = 0

    def add_edge(self, edge: Edge) -> None:
        v = edge.either()
        w = edge.other(v)
        self.adj[v].append(edge)
        self.adj[w].append(edge)
        self.E += 1

    def get_adj(self, v: int) -> List[Edge]:
        return self.adj[v]

    def get_v(self) -> int:
        return self.V

    def get_e(self) -> int:
        return self.E

    def get_edges(self) -> List[Edge]:
        result = []
        for v in range(self.V):
            for e in self.get_adj(v):
                if e.other(v) > v:
                    result.append(e)
        return result
