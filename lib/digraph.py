from __future__ import annotations
from typing import List


class Digraph:
    def __init__(self, V: int):
        self.V = V
        self.E = 0
        self.adj = [[] for _ in range(V)]

    def add_edge(self, v, w):
        self.adj[v].append(w)
        self.E += 1

    def get_adj(self, v) -> List[int]:
        return self.adj[v]

    def reverse(self) -> Digraph:
        r = Digraph(self.V)
        for v in range(self.V):
            for w in self.get_adj(v):
                r.add_edge(w, v)

        return r
