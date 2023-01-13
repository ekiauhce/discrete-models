from knapsack_cw import get_max_cost_dp, get_max_cost_brute_force

tests = [
    # вариант 1
    {
        'costs':   [4, 10, 2, 8, 10, 4, 7, 9, 10, 10],
        'weights': [5,  5, 5, 1,  9, 2, 8, 3,  1,  1],
        'capacity': 10
    },
    {
        'costs':   [4, 10, 2, 8, 10, 4, 7, 9, 10, 10],
        'weights': [7, 10, 5, 3,  8, 1, 2, 7,  6,  4],
        'capacity': 36
    },
    {
        'costs':   [4, 10,  2, 8, 10, 4, 7, 9, 10, 10],
        'weights': [4,  7, 11, 3,  8, 9, 7, 8,  7,  6],
        'capacity': 34
    },
    # вариант 2
    {
        'costs':   [ 4,  3, 10, 9,  1, 3, 5, 10, 6, 6],
        'weights': [11, 10,  2, 4, 11, 8, 7,  8, 2, 5],
        'capacity': 61
    },
    {
        'costs':   [ 4, 3, 10, 9, 1, 3, 5, 10, 6, 6],
        'weights': [10, 1,  5, 3, 3, 6, 9,  3, 9, 7],
        'capacity': 31
    },
    {
        'costs':   [ 4, 3, 10, 9, 1, 3, 5, 10, 6, 6],
        'weights': [10, 6, 10, 3, 5, 1, 5,  4, 3, 2],
        'capacity': 11
    },
    # вариант 3
    {
        'costs':   [10, 3, 2,  3,  3, 9, 2, 2, 7, 7],
        'weights': [ 6, 4, 4, 10, 10, 9, 7, 7, 3, 3],
        'capacity': 57
    },
    {
        'costs':   [10, 3, 2, 3, 3, 9, 2,  2,  7,  7],
        'weights': [ 3, 5, 4, 5, 3, 7, 5, 11, 10, 11],
        'capacity': 50
    },
    {
        'costs':   [10, 3, 2, 3, 3, 9, 2, 2, 7, 7],
        'weights': [ 3, 5, 4, 2, 8, 8, 1, 7, 4, 8],
        'capacity': 30
    },
]

for i, test in enumerate(tests, 1):
    costs = test['costs']
    weights = test['weights']
    capacity = test['capacity']

    cost_by_dp = get_max_cost_dp(costs, weights, capacity)
    cost_by_bf = get_max_cost_brute_force(costs, weights, capacity)
    has_passed = cost_by_dp == cost_by_bf

    print(f'{i}. costs={costs}, weights={weights}, capacity={capacity}')
    print('  ', 'OK' if has_passed else 'FAIL', f'cost_by_dp={cost_by_dp}, cost_by_bf={cost_by_bf}')
