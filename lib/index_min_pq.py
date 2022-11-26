
from __future__ import annotations
from typing import Generic, List, TypeVar

K = TypeVar("K") # comparable

class IndexMinPQ(Generic[K]):
    def __init__(self, capacity: int) -> None:
        self.n = 0 # number of elements on PQ
        self.pq: List[int] = [0] * (capacity + 1) # binary heap using 1-based indexing
        self.qp: List[int] = [-1] * (capacity + 1) # inverse of pq: qp[pq[i]] = pq[qp[i]] = i
        self.keys: List[K] = [None] * (capacity + 1) # keys[i] = prority of i

    def is_empty(self) -> bool:
        return self.n == 0

    def contains(self, i: int) -> bool:
        self._validate_index(i)
        return self.qp[i] != -1

    def size(self) -> int:
        return self.n

    def set(self, i: int, key: K) -> None:
        self._validate_index(i)
        if self.contains(i):
            self.keys[i] = key
            self._swim(self.qp[i])
            self._sink(self.qp[i])
        else:
            self.n += 1
            self.qp[i] = self.n
            self.pq[self.n] = i
            self.keys[i] = key
            self._swim(self.n)

    def index_on_min(self) -> int:
        self._validate_n()
        return self.pq[1]

    def min_key(self) -> K:
        return self.keys[self.index_on_min()]

    def del_min(self) -> int:
        min_index: int = self.index_on_min()
        self._exchange(1, self.n)
        self.n -= 1
        self._sink(1)
        assert min_index == self.pq[self.n+1]
        self.qp[min_index] = -1
        self.keys[min_index] = None
        return min_index

    def key_of(self, i: int) -> K:
        self._validate_index(i)
        if not self.contains(i):
            raise ValueError(f"index {i} isn't in the priority queue")
        return self.keys[i]

    def delete(self, i: int) -> None:
        self._validate_index(i)
        if not self.contains(i):
            raise ValueError(f"index {i} isn't in the priority queue")
        index: int = self.qp[i]
        self._exchange(index, self.n)
        self.n -= 1
        self._swim(index)
        self._sink(index)
        self.keys[i] = None
        self.qp[i] = -1

    def _validate_n(self) -> None:
        if self.n == 0:
            raise ValueError("Prority queue underflow")


    def _swim(self, k: int) -> None:
        while k > 1 and self._greater(k//2, k):
            self._exchange(k, k//2)
            k = k//2

    def _sink(self, k: int) -> None:
        while 2*k <= self.n:
            j = 2*k
            if j < self.n and self._greater(j, j+1):
                j += 1
            if not self._greater(k, j):
                break
            self._exchange(k, j)
            k = j

    def _greater(self, i: int, j: int) -> bool:
        return self.keys[self.pq[i]] > self.keys[self.pq[j]]


    def _exchange(self, i: int, j: int) -> None:
        self.pq[i], self.pq[j] = self.pq[j], self.pq[i]
        self.qp[self.pq[i]] = i
        self.qp[self.pq[j]] = j

    def _validate_index(self, i: int) -> None:
        if i < 0:
            raise ValueError(f"index is negative: {i}")
        if i >= len(self.pq)-1:
            raise ValueError(f"index >= capacity: {i}")
