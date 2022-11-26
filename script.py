#!/usr/bin/env python3
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from typing import Dict, List, Set, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import math
from lib.digraph import Digraph
from lib.dijkstra_sp import DijkstraSP
from lib.directed_edge import DirectedEdge
from lib.edge_weighted_digraph import EdgeWeightedDigraph
from lib.edge_weighted_graph import EdgeWeightedGraph
from lib.edge import Edge
from lib.kosaraju_scc import KosarajuSCC
from lib.mst import KruskalMst, PrimMst
import itertools
from tabulate import tabulate


parser = argparse.ArgumentParser()
parser.add_argument('--id', default=1, type=int, choices=[1, 2, 3, 4, 5, 6])
subparser = parser.add_subparsers(dest='command', required=True)

draw_parser = subparser.add_parser('draw')
draw_parser.add_argument('-l', '--layout', choices=['circular', 'canonic'], default='circular')
draw_parser.add_argument('--by', choices=['adj', 'inc'], default='adj')
draw_parser.add_argument('--obj', default='graph', choices=['graph', 'cond_graph', 'weighted_graph'])

subparser.add_parser('test')

count_parser = subparser.add_parser('count')
group = count_parser.add_mutually_exclusive_group(required=True)
group.add_argument('--indegree', help="Полустепени захода", action='store_true')
group.add_argument('--outdegree', help="Полустепени исхода", action='store_true')

scc_parser = subparser.add_parser('scc')
base_parser = subparser.add_parser('base')
indep_sets_parser = subparser.add_parser('indep_sets')
color_parser = subparser.add_parser('color')
mst_parser = subparser.add_parser('mst')
mst_parser.add_argument('--algorithm', choices=['prim', 'kruskal'], required=True)

sp_parser = subparser.add_parser('sp')
sp_parser.add_argument('--start-vertex', type=int, required=True)

ADJ = 'adjacency_matrix%s'
INC = 'incidence_matrix%s'
REACH = 'reachability_matrix%s'
COND_ADJ = 'condensation_adjacency_matrix%s'
WEIGHTED_EDGES = 'weighted_edges%s'


def main(args):
    if args.command == 'draw':
        draw_a_graph(args.obj, args.by, args.id, args.layout)
    elif args.command == 'test':
        if args.id == 1:
            adjacency_matrix = get_adjacency_matrix_by_incidence(read_matrix(INC % args.id))
            write_matrix(ADJ % args.id, adjacency_matrix)

            incidence_matrix = get_incidence_matrix_by_adjacency(read_matrix(ADJ % args.id))
            write_matrix(INC % args.id, incidence_matrix)
        elif args.id == 2:
            adj_matrix = read_matrix(ADJ % args.id)
            reachability_matrix = get_reachability_matrix_by_adjacency(adj_matrix)
            write_matrix(REACH % args.id, reachability_matrix)

            scc: List[Set[int]] = get_scc_by_adjacency_matrix(adj_matrix)
            condensation_adj_matrix = get_adjacency_matrix_by_scc_and_reachability(scc, reachability_matrix)
            write_matrix(COND_ADJ % args.id, condensation_adj_matrix)

    elif args.command == 'count':
        if args.indegree:
            indegrees = get_indegrees(read_matrix(ADJ % args.id))
            for i, indegree in enumerate(indegrees):
                print(f'Полустепени захода x{i+1} = {indegree}')
        elif args.outdegree:
            indegrees = get_outdegrees(read_matrix(ADJ % args.id))
            for i, indegree in enumerate(indegrees):
                print(f'Полустепени исхода x{i+1} = {indegree}')
    elif args.command == 'scc':
        adj_matrix = read_matrix(ADJ % args.id)
        reachability_matrix = get_reachability_matrix_by_adjacency(adj_matrix)

        scc: List[Set[int]] = get_scc_by_adjacency_matrix(adj_matrix)

        print("Сильные компоненты:")
        print(json.dumps(
            {component+1 : [n+1 for n in nodes] for component, nodes in enumerate(scc)},
            indent=4
        ))
    elif args.command == 'mst':
        edges = read_weighted_edges(WEIGHTED_EDGES % args.id)
        vertices: Set[int] = set()
        for e in edges:
            vertices.add(e[0])
            vertices.add(e[1])
        weighted_graph = EdgeWeightedGraph(len(vertices))
        for e in edges:
            weighted_graph.add_edge(Edge(*e))
        mst = PrimMst(weighted_graph) if args.algorithm == 'prim' else KruskalMst(weighted_graph)
        mst_edges = mst.get_edges()

        nx_graph = nx.Graph()
        for e in mst_edges:
            v = e.either()
            w = e.other(v)
            nx_graph.add_edge(v+1, w+1, weight=e.get_weight())
        pos = nx.circular_layout(nx_graph)
        opts = {
            "node_color": "white",
            "edgecolors": "black",
        }
        nx.draw_networkx(nx_graph, pos, **opts)
        edge_labels = nx.get_edge_attributes(nx_graph, "weight")
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    elif args.command == 'sp':
        edges = read_weighted_edges(WEIGHTED_EDGES % args.id)
        vertices: Set[int] = set()
        for e in edges:
            vertices.add(e[0])
            vertices.add(e[1])
        weighted_digraph = EdgeWeightedDigraph(len(vertices))
        for e in edges:
            weighted_digraph.add_edge(DirectedEdge(*e))
        sp = DijkstraSP(weighted_digraph, args.start_vertex-1)
        for v in sorted(list(vertices)):
            print(f"Path to {v+1} is {sp.get_path_to(v)}. Total weight is {sp.get_dist_to(v)}")


    elif args.command == 'base':
        adj_matrix = read_matrix(ADJ % args.id)
        reachability_matrix = get_reachability_matrix_by_adjacency(adj_matrix)
        scc = get_scc_by_adjacency_matrix(adj_matrix)

        condensation_adj_matrix = get_adjacency_matrix_by_scc_and_reachability(scc, reachability_matrix)

        condensation_base = [i for i, v in enumerate(get_indegrees(condensation_adj_matrix)) if v == 0]
        print("Базы графа:")
        components = [component for i, component in enumerate(scc) if i in condensation_base]
        for base in itertools.product(*components):
            print(base)

    elif args.command == 'indep_sets':
        adj_matrix = read_matrix(ADJ % args.id)
        graph: Dict[int, Set[int]] = get_graph_as_dict(adj_matrix)

        result = set()
        get_max_independent_sets(graph, result)

        result = list(
            map(
                lambda s: set([x+1 for x in s]),
                sorted(list(result), reverse=True, key=lambda x: len(x))
            )
        )
        print("Все независимые множества графа:")
        for s in result:
            print(s)

        result = result if result else [[]]
        print(f"Число независимости: {len(next(iter(result)))}")
    elif args.command == 'color':
        graph = get_graph_as_dict(read_matrix(ADJ % args.id))
        graph: Dict[int, Set[int]] = dict(sorted(graph.items(), key=lambda x: len(x[1]), reverse=True))

        colors = [-1 for _ in range(len(graph))]
        color = 0
        not_colored = list(graph.keys())

        while not_colored:
            v = not_colored.pop(0)
            color += 1
            colors[v] = color
            not_adj = list(filter(lambda u: u not in graph[v], not_colored))
            for u in not_adj:
                if u in not_colored and all(colors[w] != color for w in graph[u]):
                    colors[u] = color
                    not_colored.remove(u)

        print(tabulate([[v+1, c] for v, c in enumerate(colors)], headers=['Вершина', 'Цвет']))
        draw_a_graph('graph', 'adj', args.id, 'circular', colors)

def get_graph_as_dict(adj_matrix) -> Dict[int, Set[int]]:
    graph: Dict[int, Set[int]] = {}
    for u in range(len(adj_matrix)):
        for v in range(len(adj_matrix)):
            if adj_matrix[u][v]:
                if u not in graph:
                    graph[u] = set()
                graph[u].add(v)
    return graph


def get_max_independent_sets(graph: Dict[int, Set[int]], result: Set[Set[int]]):
    if all(map(lambda adj: len(adj) == 0, graph.values())):
        result.add(frozenset(graph.keys()))
        return

    graph_copy: Dict[int, Set[int]] = deepcopy(graph)

    node = next(iter(graph.keys()))
    adj = set(graph[node])
    del graph[node]
    for u in adj:
        graph[u].remove(node)

    # left branch
    get_max_independent_sets(graph, result)

    it = iter(graph_copy.keys())
    node = next(it)
    while not graph_copy[node]:
        node = next(it)

    adj = graph_copy[node]
    graph_copy[node] = set()

    for u in adj:
        graph_copy[u].remove(node)
        adj_adj = graph_copy[u]
        del graph_copy[u]
        for v in adj_adj:
            graph_copy[v].remove(u)

    # right branch
    get_max_independent_sets(graph_copy, result)


def get_scc_by_adjacency_matrix(adj_matrix) -> List[Set[int]]:
    digraph = Digraph(len(adj_matrix))
    for v, w in get_edges_list(adj_matrix):
        digraph.add_edge(v, w)

    return KosarajuSCC(digraph).get_scc()



def get_adjacency_matrix_by_scc_and_reachability(scc, reachability_matrix):
    condensation_adj_matrix = [[0 for _ in range(len(scc))] for _ in range(len(scc))]
    for i in range(len(scc)):
        for j in range(i+1, len(scc)):
            # выбираем произвольную ноду, так как если хоть одна верншина из компонента связности scc[i]
            # окажется соединенной с любой нодой из scc[j], то и любая другая вершина из scc[i] будет
            # транзитивно соединена с остальными из scc[j]
            v = next(iter(scc[i]))
            w = next(iter(scc[j]))
            condensation_adj_matrix[i][j] = reachability_matrix[v][w]
            condensation_adj_matrix[j][i] = reachability_matrix[w][v]

    return condensation_adj_matrix


# Floyd–Warshall algorithm, O(|V|^3)
def get_reachability_matrix_by_adjacency(adjacency_matrix: List[List[int]]):
    n = len(adjacency_matrix)
    edges = get_edges_list(adjacency_matrix)
    dist = [[math.inf for _ in range(n)] for _ in range(n)]
    for u, v in edges:
        dist[u][v] = 1

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    reachability_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if dist[i][j] < math.inf:
                reachability_matrix[i][j] = 1

    return reachability_matrix


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
                edges.append((i, j))

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


def draw_a_graph(object: str, by: str, id: int, layout: str, colors=[]):
    opts = {
        "node_color": "white",
        "edgecolors": "black",
    }

    if object == 'graph':
        if by == 'adj':
            edges = get_edges_list(read_matrix(ADJ % id))
        elif by == 'inc':
            edges = get_edges_list(get_adjacency_matrix_by_incidence(read_matrix(INC % id)))
    elif object == 'cond_graph':
        adj_matrix = read_matrix(ADJ % id) # NOTE: для cond_graph безусловно считываем только из adj matrix
        reachability_matrix = get_reachability_matrix_by_adjacency(adj_matrix)
        scc: List[Set[int]] = get_scc_by_adjacency_matrix(adj_matrix)

        condensation_adj_matrix = get_adjacency_matrix_by_scc_and_reachability(scc, reachability_matrix)
        edges = get_edges_list(condensation_adj_matrix)
    elif object == 'weighted_graph':
        edges = read_weighted_edges(WEIGHTED_EDGES % id)

        nx_graph = nx.Graph()
        for e in edges:
            nx_graph.add_edge(e[0]+1, e[1]+1, weight=e[2])
        pos = nx.circular_layout(nx_graph)
        nx.draw_networkx(nx_graph, pos, **opts)
        edge_labels = nx.get_edge_attributes(nx_graph, "weight")
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels, label_pos=0.65)

        ax = plt.gca()
        ax.margins(0.04)
        plt.axis("off")
        # plt.tight_layout()
        plt.show()
        return

    edges = [ (u+1, v+1) for u, v in edges]
    G = nx.DiGraph([ (f"x{u}", f"x{v}") for u, v in edges])
    edge_labels = {(f"x{u}", f"x{v}"): f"a{i}" for i, (u, v) in enumerate(edges, 1)}
    pos = get_pos(G, layout)

    if colors:
        opts['cmap'] = plt.get_cmap("Set3")
        colors = {f"x{i}" : c for i, c in enumerate(colors, 1) }
        opts["node_color"] = [colors[v] for v in G.nodes]

    nx.draw_networkx(G, pos, **opts)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.60)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()


def read_matrix(name: str) -> List[List[int]]:
    with open(f"{name}.in", "r") as f:
        return [list(map(int, line.rstrip('\n').split())) for line in f.readlines()]

def read_weighted_edges(name: str) -> List[Tuple[int, int, float]]:
    with open(f"{name}.in", "r") as f:
        result = []
        for line in f.readlines():
            v, w, weight = line.rstrip('\n').split()
            result.append((int(v)-1, int(w)-1, float(weight)))
        return result

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