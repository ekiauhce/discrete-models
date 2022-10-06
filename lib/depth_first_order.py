from __future__ import annotations
from .digraph import Digraph


class DepthFirstOrder:
    def __init__(self, G: Digraph) -> None:
        self.marked = [False for _ in range(G.V)]
        self.pre = []
        self.post = []
        for v in range(G.V):
            if not self.marked[v]:
                self.dfs(G, v)

    def dfs(self, G: Digraph, v: int) -> None:
        self.marked[v] = True
        self.pre.append(v)

        for w in G.get_adj(v):
            if not self.marked[w]:
                self.dfs(G, w)

        self.post.append(v)

    def get_reverse_post(self):
        return reversed(self.post)
