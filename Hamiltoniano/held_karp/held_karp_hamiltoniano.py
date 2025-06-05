import networkx as nx
import numpy as np
import itertools

def held_karp_tsp(graph):
    nodes = list(graph.nodes)
    n = len(nodes)
    if n <= 1:
        return float('inf'), []

    node_index = {node: i for i, node in enumerate(nodes)}

    dist = np.full((n, n), float('inf'))
    for u, v, data in graph.edges(data=True):
        i, j = node_index[u], node_index[v]
        weight = data.get('weight', 1)
        dist[i][j] = weight
        dist[j][i] = weight

    dp = {}
    parent = {}

    for k in range(1, n):
        dp[(1 << k, k)] = dist[0][k]
        parent[(1 << k, k)] = 0

    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = sum(1 << bit for bit in subset)
            for k in subset:
                prev_bits = bits & ~(1 << k)
                options = []
                for m in subset:
                    if m == k:
                        continue
                    if (prev_bits, m) in dp:
                        cost = dp[(prev_bits, m)] + dist[m][k]
                        options.append((cost, m))
                if options:
                    dp[(bits, k)], parent[(bits, k)] = min(options)

    bits = (1 << n) - 2
    options = []
    for k in range(1, n):
        if (bits, k) in dp:
            options.append((dp[(bits, k)] + dist[k][0], k))
    if not options:
        return float('inf'), []

    opt_cost, parent_node = min(options)

    path = [0]
    last = parent_node
    bits = (1 << n) - 2
    for _ in range(n - 1):
        path.append(last)
        bits, last = bits & ~(1 << last), parent[(bits, last)]
    path.append(0)
    node_path = [nodes[i] for i in reversed(path)]

    return opt_cost, node_path

grafos = [
    [('A','B'), ('B','C'), ('C','D'), ('D','A')],
    [('A','B'), ('B','C'), ('C','D')],
    [('A','B'), ('A','C'), ('B','C'), ('C','D'), ('D','E')],
    [('A','B'), ('B','C'), ('C','D'), ('D','E'), ('E','A')],
    [('A','B'), ('A','C'), ('A','D'), ('A','E')],
    [('A','B'), ('B','C'), ('C','D'), ('D','E'), ('E','F'), ('F','A')],
    [('A','B'), ('B','C'), ('C','E'), ('E','D')],
    [('A','B'), ('B','C'), ('C','D'), ('D','A'), ('A','C')],
    [('A','B'), ('B','C'), ('C','A'), ('C','D'), ('D','E')],
    [('A','B'), ('B','C'), ('C','D'), ('D','E'), ('E','F'), ('F','A'), ('B','E'), ('C','F')]
]

for i, edges in enumerate(grafos, 1):
    G = nx.Graph()
    G.add_edges_from([(u, v, {'weight': 1}) for u, v in edges])

    print(f"\n--- Grafo {i} ---")
    cost, path = held_karp_tsp(G)
    if cost == float('inf'):
        print("Não é possível formar um ciclo Hamiltoniano.")
    else:
        print("Custo mínimo:", cost)
        print("Caminho ótimo:", " → ".join(path))
