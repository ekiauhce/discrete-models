from __future__ import annotations
from functools import total_ordering

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
