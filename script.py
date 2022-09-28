#!/usr/bin/env python3

import argparse
from typing import List
import networkx as nx
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='command', required=True)

draw_parser = subparser.add_parser('draw')
draw_parser.add_argument('-l', '--layout', choices=['circular', 'canonic'], default='canonic')
draw_parser.add_argument('--by', choices=['adj', 'inc'], default='adj')

subparser.add_parser('test')

count_parser = subparser.add_parser('count')
group = count_parser.add_mutually_exclusive_group(required=True)
group.add_argument('--indegree', help="Полустепени захода", action='store_true')
group.add_argument('--outdegree', help="Полустепени исхода", action='store_true')

ADJ = 'adjacency_matrix'
INC = 'incidence_matrix'


def main(args):
    if args.command == 'draw':
        draw_a_graph(args.by)
    elif args.command == 'test':
        adjacency_matrix = get_adjacency_matrix_by_incidence(read_matrix(INC))
        write_matrix(ADJ, adjacency_matrix)

        incidence_matrix = get_incidence_matrix_by_adjacency(read_matrix(ADJ))
        write_matrix(INC, incidence_matrix)
    elif args.command == 'count':
        if args.indegree:
            indegrees = get_indegrees(read_matrix(ADJ))
            for i, indegree in enumerate(indegrees):
                print(f'Полустепени захода x{i+1} = {indegree}')
        elif args.outdegree:
            indegrees = get_outdegrees(read_matrix(ADJ))
            for i, indegree in enumerate(indegrees):
                print(f'Полустепени исхода x{i+1} = {indegree}')


def get_adjacency_matrix_by_incidence(incidence_matrix: List[List[int]]):
    if not incidence_matrix:
        raise ValueError("incidence matrix must be non empty")

    edges = [[-1, -1] for _ in range(len(incidence_matrix[0]))]

    for i, row in enumerate(incidence_matrix):
        for j, v in enumerate(row):
            if v == 1:
                edges[j][0] = i
            elif v == -1:
                edges[j][1] = i

    n = len(incidence_matrix)
    adj_matrix = [ [0 for _ in range(n)] for _ in range(n) ]

    for u, v in edges:
        adj_matrix[u][v] = 1

    return adj_matrix


def get_incidence_matrix_by_adjacency(adjacency_matrix: List[List[int]]):
    if not adjacency_matrix:
        raise ValueError("adjacency matrix must be non empty")
    nodes_count = len(adjacency_matrix)
    edges_count = 0
    for row in adjacency_matrix:
        for v in row:
            if v == 1:
                edges_count += 1

    inc_matrix = [[0 for _ in range(edges_count)] for _ in range(nodes_count) ]

    edges_visited = 0
    for i, row in enumerate(adjacency_matrix):
        for j, v in enumerate(row):
            if v == 1:
                inc_matrix[i][edges_visited] = 1
                inc_matrix[j][edges_visited] = -1
                edges_visited += 1

    return inc_matrix


def get_edges_list(adjacency_matrix: List[List[int]]):
    edges = []
    for i, row in enumerate(adjacency_matrix):
        for j, v in enumerate(row):
            if v == 1:
                edges.append((i+1, j+1))

    return edges


def get_indegrees(adjacency_matrix: List[List[int]]):
    indegrees = [ 0 for _ in range(len(adjacency_matrix)) ]

    for i, row in enumerate(adjacency_matrix):
        for j, v in enumerate(row):
            if v == 1:
                indegrees[j] += 1

    return indegrees


def get_outdegrees(adjacency_matrix: List[List[int]]):
    outdegrees = [ 0 for _ in range(len(adjacency_matrix)) ]

    for i, row in enumerate(adjacency_matrix):
        for _, v in enumerate(row):
            if v == 1:
                outdegrees[i] += 1

    return outdegrees


def draw_a_graph(by: str):
    if by == 'adj':
        edges = get_edges_list(read_matrix(ADJ))
    elif by == 'inc':
        edges = get_edges_list(get_adjacency_matrix_by_incidence(read_matrix(INC)))

    G = nx.DiGraph([ (f"x{u}", f"x{v}") for u, v in edges])
    edge_labels = {(f"x{u}", f"x{v}"): f"a{i}" for i, (u, v) in enumerate(edges, 1)}
    pos = get_pos(G, args.layout)
    nx.draw_networkx(G, pos, **{"node_color": "white", "edgecolors": "black",})
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.65)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()


def read_matrix(name: str) -> List[List[int]]:
    with open(f"{name}.in", "r") as f:
        return [list(map(int, line.rstrip('\n').split())) for line in f.readlines()]

def write_matrix(name: str, matrix: List[List[int]]) -> None:
    with open(f"{name}.out", 'w') as f:
        for row in matrix:
            f.write(" ".join(["%2d" % v for v in row]))
            f.write("\n")

def get_pos(G, layout: str):
    if layout == 'canonic':
        groups = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8]
        ]
        pos = {}
        for i, group in enumerate(groups):
            pos.update({f"x{n}": (i, j + 0.5 * i) for j, n in enumerate(group)})
        return pos
    elif layout == 'circular':
        return nx.circular_layout(G)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)