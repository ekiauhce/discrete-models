from __future__ import annotations
from typing import List
from functools import total_ordering


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


@total_ordering
class DirectedEdge:
    def __init__(self, v: int, w: int, weight: float) -> None:
        self.v = v
        self.w = w
        self.weight = weight

    def start(self) -> int:
        return self.v

    def end(self) -> int:
        return self.w

    def __eq__(self, that):
        return isinstance(that, DirectedEdge) and self.weight == that.weight

    def __lt__(self, that):
        return isinstance(that, DirectedEdge) and self.weight < that.weight

    def __repr__(self) -> str:
        return f'({self.v+1}->{self.w+1}, {self.weight})'


class EdgeWeightedDirectedCycle:
    def __init__(self, G: EdgeWeightedDigraph) -> None:
        self.marked: List[bool] = [False] * G.V
        self.on_stack: List[bool] = [False] * G.V
        self.edge_to: List[DirectedEdge] = [None] * G.V
        self.cycle: List[DirectedEdge] = None

        for v in range(G.V):
            if not self.marked[v]:
                self._dfs(G, v)

    def _dfs(self, G: EdgeWeightedDigraph, v: int) -> None:
        self.on_stack[v] = True
        self.marked[v] = True
        for edge in G.get_adj(v):
            w: int = edge.end()
            if self.cycle:
                return
            elif not self.marked[w]:
                self.edge_to[w] = edge
                self._dfs(G, w)
            elif self.on_stack[w]:
                self.cycle = []
                f: DirectedEdge = edge
                while f.start() != w:
                    self.cycle.append(f)
                    f = self.edge_to[f.start()]
                self.cycle.append(f)
                return
        self.on_stack[v] = False

    def has_cycle(self) -> bool:
        return bool(self.cycle)
