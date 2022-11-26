from __future__ import annotations
from typing import List

from lib.directed_edge import DirectedEdge

class EdgeWeightedDigraph:
    def __init__(self, V: int) -> None:
        self.V: int = V
        self.E: int = 0
        self.adj = [[] for _ in range(V)]

    def add_edge(self, e: DirectedEdge) -> None:
        self.adj[e.start()].append(e)
        self.E += 1

    def get_adj(self, v: int) -> List[DirectedEdge]:
        return self.adj[v]

    def get_edges(self) -> List[DirectedEdge]:
        result = []
        for v in range(self.V):
            for e in self.get_adj(v):
                result.append(e)
        return result

    def get_v(self) -> int:
        return self.V

    def get_e(self) -> int:
        return self.E