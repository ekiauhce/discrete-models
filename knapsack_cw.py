from typing import List
from random import randint
import time
import statistics
import numpy

def get_max_cost_brute_force(
    costs: List[float], weights: List[float], capacity: float
) -> float:
    assert len(costs) == len(weights)
    items_count = len(costs)

    max_cost = -float('inf')
    for state in range(1 << items_count):
        bits = list(
            map(
                lambda x: 1 if state & x else 0, map(
                    lambda x: 1 << x, reversed(range(items_count)))
            )
        )
        cost = sum(costs[i] for i, t in enumerate(bits) if t)
        weight = sum(weights[i] for i, t in enumerate(bits) if t)
        if weight <= capacity and cost > max_cost:
            max_cost = cost

    return max_cost


def get_max_cost_dp(
    costs: List[float], weights: List[float], capacity: float
) -> float:
    assert len(costs) == len(weights)
    items_count = len(costs)
    # каждый элемент матрицы хранит значение
    # максимальной ценности при том, что
    # - есть выбор из предметов от 0 до i
    # - рюкзак объема j
    dp = numpy.zeros((items_count+1, capacity+1))

    for i in range(items_count):
        for j in range(1, capacity+1):
            # если вес i-ого предмета больше объема j
            if weights[i] > j:
                # считаем оптимальным вариантом результат
                # для предыдущего предмета (строки) при
                # том же объеме j
                dp[i][j] = dp[i-1][j]
            else: # иначе i-ый предмет вмещается
                dp[i][j] = max(
                    # класть i-ый предмет в рюкзак не выгодно
                    dp[i-1][j],
                    # кладем i-ый предмет в рюкзак
                    costs[i] + dp[i-1][j - weights[i]]
                )

    # в крайнем справа снизу (в этой клетке мы рассматриваем
    # все доступные предметы и макс. допустимый объем рюкзака)
    # элементе матрицы dp будет ответ - наш экстремум
    return dp[items_count-1][capacity]


def get_dp_bits(
    weights: List[int], capacity: int, dp: List[List[int]]
) -> List[int]:
    items_count = len(weights)
    bits = []
    for i in range(items_count-1, -1, -1):
        if dp[i][capacity] != dp[i-1][capacity]:
            bits.append(1)
            capacity -= weights[i]
        else:
            bits.append(0)
    bits = list(reversed(bits))
    return bits

def nanos_to_ms(nanos):
    return nanos / 1_000_000

if __name__ == '__main__':
import matplotlib.pyplot as plt

RETRIES = 5
LO_ITEMS_COUNT = 100
HI_ITEMS_COUNT = 1001
ITEMS_COUNT_STEP = 100
LO_WEIGHT = LO_COST = 1
HI_WEIGHT = HI_COST = 15
LO_CAPACITY = 20
HI_CAPACITY = 51
CAPACITY_STEP = 10

x_values = []
y_values_per_capacity = {}
for items_count in range(
    LO_ITEMS_COUNT, HI_ITEMS_COUNT, ITEMS_COUNT_STEP):
    x_values.append(items_count)
    weights = [
        randint(LO_WEIGHT, HI_WEIGHT) for _ in range(items_count)
    ]
    costs = [
        randint(LO_COST, HI_COST) for _ in range(items_count)
    ]

    for capacity in range(LO_CAPACITY, HI_CAPACITY, CAPACITY_STEP):
        elapsed_time = []
        for _ in range(RETRIES):
            start = time.time_ns()
            get_max_cost_dp(costs, weights, capacity)
            elapsed_time.append(
                nanos_to_ms(time.time_ns() - start)
            )
        y_values_per_capacity.setdefault(capacity, []). \
            append(statistics.median(elapsed_time))

for capacity, y_values in y_values_per_capacity.items():
    plt.plot(x_values, y_values, label=f"capacity={capacity}")
plt.xlabel("number of items")
plt.ylabel("ms")
plt.legend(loc="upper left")
plt.show()
