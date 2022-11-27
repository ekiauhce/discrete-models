from __future__ import annotations
from typing import List

from .edge_weighted_digraph import EdgeWeightedDigraph, DirectedEdge, EdgeWeightedDirectedCycle
from .index_min_pq import IndexMinPQ

class DijkstraSP:
    def __init__(self, G: EdgeWeightedDigraph, s: int) -> None:
        self.edge_to: List[DirectedEdge] = [None] * G.V
        self.dist_to: List[float] = [float('inf')] * G.V
        self.dist_to[s] = 0.0

        self.pq: IndexMinPQ = IndexMinPQ(G.V)
        self.pq.set(s, 0.0)
        while not self.pq.is_empty():
            self._relax(G, self.pq.del_min())

    def has_path_to(self, v: int) -> bool:
        return self.dist_to[v] < float('inf')

    def get_dist_to(self, v: int) -> float:
        return self.dist_to[v]

    def get_path_to(self, v: int) -> List[DirectedEdge]:
        if not self.has_path_to(v):
            return []
        path = []
        e = self.edge_to[v]
        while e is not None:
            path.insert(0, e)
            e = self.edge_to[e.start()]
        return path

    def _relax(self, G: EdgeWeightedDigraph, v: int) -> None:
        for edge in G.get_adj(v):
            w: int = edge.end()
            if self.dist_to[w] > self.dist_to[v] + edge.weight:
                self.dist_to[w] = self.dist_to[v] + edge.weight
                self.edge_to[w] = edge
                self.pq.set(w, self.dist_to[w])


class BellmanSP:
    EPSILON = 1E-14

    def __init__(self, G: EdgeWeightedDigraph, s: int) -> None:
        self.edge_to: List[DirectedEdge] = [None] * G.V
        self.dist_to: List[float] = [float('inf')] * G.V
        self.dist_to[s] = 0.0
        self.cycle: List[DirectedEdge] = None
        self.cost = 0

        self.on_queue: List[bool] = [False] * G.V
        self.queue: List[int] = []
        self.queue.append(s)
        self.on_queue[s] = True
        while self.queue and not self._has_negative_cycle():
            v: int = self.queue.pop(0)
            self.on_queue[v] = False
            self._relax(G, v)

    def get_dist_to(self, v: int) -> float:
        if self._has_negative_cycle():
            raise ValueError("Has negative cycle")
        return self.dist_to[v]

    def get_path_to(self, v: int) -> List[DirectedEdge]:
        if not self.has_path_to(v):
            return []
        path = []
        e = self.edge_to[v]
        while e is not None:
            path.insert(0, e)
            e = self.edge_to[e.start()]
        return path

    def has_path_to(self, v: int) -> bool:
        return self.dist_to[v] < float('inf')

    def _has_negative_cycle(self) -> bool:
        return bool(self.cycle)

    def _relax(self, G: EdgeWeightedDigraph, v: int) -> None:
        for edge in G.get_adj(v):
            w: int = edge.end()
            if self.dist_to[w] > self.dist_to[v] + edge.weight + BellmanSP.EPSILON:
                self.dist_to[w] = self.dist_to[v] + edge.weight
                self.edge_to[w] = edge
                if not self.on_queue[w]:
                    self.queue.append(w)
                    self.on_queue[w] = True
            self.cost += 1
            if self.cost % G.V == 0:
                self._find_negative_cycle()
                if self._has_negative_cycle():
                    return

    def _find_negative_cycle(self):
        V: int = len(self.edge_to)
        spt = EdgeWeightedDigraph(V)
        for v in range(V):
            if self.edge_to[v] != None:
                spt.add_edge(self.edge_to[v])
        finder = EdgeWeightedDirectedCycle(spt)
        self.cycle = finder.cycle


