from __future__ import annotations
from typing import List
from lib.edge_weighted_graph import Edge
from lib.edge_weighted_graph import EdgeWeightedGraph
import heapq

from lib.union_find import UnionFind

class PrimMst:
    def __init__(self, graph: EdgeWeightedGraph) -> None:
        self.marked = [False for _ in range(graph.get_v())]
        self.mst: List[Edge] = []
        self.pq: List[Edge] = []
        self.total = 0.

        self.visit(graph, 0)
        while self.pq:
            edge = heapq.heappop(self.pq)
            v = edge.either()
            w = edge.other(v)

            if self.marked[v] and self.marked[w]:
                continue

            self.mst.append(edge)
            self.total += edge.get_weight()
            if not self.marked[v]:
                self.visit(graph, v)
            if not self.marked[w]:
                self.visit(graph, w)

    def get_edges(self) -> List[Edge]:
        return self.mst

    def get_weight(self):
        return self.total

    def visit(self, graph: EdgeWeightedGraph, v: int) -> None:
        self.marked[v] = True
        for edge in graph.get_adj(v):
            if not self.marked[edge.other(v)]:
                heapq.heappush(self.pq, edge)

class KruskalMst:
    def __init__(self, graph: EdgeWeightedGraph) -> None:
        self.mst: List[Edge] = []
        self.pq: List[Edge] = []
        for edge in graph.get_edges():
            heapq.heappush(self.pq, edge)
        uf = UnionFind(graph.get_v())

        while self.pq and len(self.mst) < graph.get_v() - 1:
            edge = heapq.heappop(self.pq)
            v = edge.either()
            w = edge.other(v)
            if uf.connected(v, w):
                continue
            uf.union(v, w)
            self.mst.append(edge)

    def get_edges(self) -> List[Edge]:
        return self.mst
