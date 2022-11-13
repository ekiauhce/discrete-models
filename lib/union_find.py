class UnionFind:
    def __init__(self, N: int) -> None:
        self.id = [i for i in range(N)]
        self.sz = [1 for i in range(N)]

    def find(self, p: int) -> int:
        while p != self.id[p]:
            p = self.id[p]
        return p

    def union(self, p: int, q: int) -> None:
        p_root: int = self.find(p)
        q_root: int = self.find(q)

        if p_root == q_root:
            return

        if self.sz[p_root] < self.sz[q_root]:
            self.id[p_root] = q_root
            self.sz[q_root] += self.sz[p_root]
        else:
            self.id[q_root] = p_root
            self.sz[p_root] += self.sz[q_root]

    def connected(self, p: int, q: int) -> bool:
        return self.find(p) == self.find(q)
