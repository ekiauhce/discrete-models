from __future__ import annotations
from typing import List

from lib.directed_edge import DirectedEdge
from lib.edge_weighted_digraph import EdgeWeightedDigraph
from lib.index_min_pq import IndexMinPQ

class DijkstraSP:
    def __init__(self, G: EdgeWeightedDigraph, s: int) -> None:
        self.edge_to: List[DirectedEdge] = [None] * G.get_v()
        self.dist_to: List[float] = [float('inf')] * G.get_v()
        self.dist_to[s] = 0.0

        self.pq: IndexMinPQ = IndexMinPQ(G.get_v())
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
