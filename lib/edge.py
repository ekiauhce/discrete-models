from functools import total_ordering

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
