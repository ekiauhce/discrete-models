from __future__ import annotations
from functools import total_ordering
from typing import List

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


@total_ordering
class Edge:
    def __init__(self, v: int, w: int, weight: float) -> None:
        self.v = v
        self.w = w
        self.weight = weight

    def either(self) -> int:
        return self.v

    def other(self, vertex: int) -> int:
        if vertex == self.v:
            return self.w
        elif vertex == self.w:
            return self.v
        else:
            raise ValueError(f"Got inconsistent vertex {vertex}")

    def get_weight(self) -> float:
        return self.weight

    def __eq__(self, that):
        return isinstance(that, Edge) and self.weight == that.weight

    def __lt__(self, that):
        return isinstance(that, Edge) and self.weight < that.weight

    def __repr__(self) -> str:
        return f'({self.v}, {self.w}, {self.weight})'
