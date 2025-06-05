import networkx as nx
import itertools
import numpy as np
import matplotlib.pyplot as plt
import time
from python_tsp.heuristics import solve_tsp_local_search

# --- Funções comuns ---
def draw_graph(graph, graph_id, folder, title="", highlight_path=None, edge_color='red'):
    plt.figure(figsize=(6, 6))
    plt.title(f"{title} - Grafo {graph_id}")
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue',
            node_size=500, edge_color='gray', font_weight='bold')

    if highlight_path:
        edges = list(zip(highlight_path, highlight_path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=edge_color, width=2)

    plt.savefig(f"{folder}/grafo_{folder.lower()}_{graph_id}.png")
    plt.close()

def graph_to_tsp_matrix(graph):
    nodes = list(graph.nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    tsp_matrix = np.full((n, n), float('inf'))

    for u, v, data in graph.edges(data=True):
        i, j = node_index[u], node_index[v]
        weight = data.get('weight', 1)
        tsp_matrix[i][j] = weight
        tsp_matrix[j][i] = weight

    return tsp_matrix, nodes

def format_path(path):
    return " → ".join(path)

# --- Algoritmo Brute Force ---
def is_hamiltonian(graph):
    nodes = list(graph.nodes)
    n = len(nodes)

    for perm in itertools.permutations(nodes):
        is_cycle = True
        for i in range(n):
            if not graph.has_edge(perm[i], perm[(i + 1) % n]):
                is_cycle = False
                break
        if is_cycle:
            return True, perm + (perm[0],)
    return False, None

def run_bruteforce(edges, graph_id, print_details=True):
    G = nx.Graph()
    G.add_edges_from(edges)
    if print_details:
        print(f"\n--- BruteForce Grafo {graph_id} ---")

    start = time.time()
    is_ham, cycle = is_hamiltonian(G)
    end = time.time()

    if print_details:
        print("É Hamiltoniano?", is_ham)
        if is_ham:
            print("Ciclo Hamiltoniano:", cycle)
            print("Caminho formatado:", format_path(cycle))
        tsp_matrix, nodes = graph_to_tsp_matrix(G)
        print("\nMatriz TSP:")
        print("   ", "  ".join(nodes))
        for i, row in enumerate(tsp_matrix):
            row_str = "  ".join(f"{val if val != float('inf') else '-':>3}" for val in row)
            print(f"{nodes[i]}  {row_str}")
        draw_graph(G, graph_id, "forca_bruta", "BruteForce", cycle, edge_color='red')

    return end - start

# --- Algoritmo Christofides ---
def christofides_tsp(graph):
    T = nx.minimum_spanning_tree(graph)
    odd_degree_nodes = [v for v in T.nodes if T.degree(v) % 2 == 1]
    G_odd = graph.subgraph(odd_degree_nodes)
    matching = nx.algorithms.matching.min_weight_matching(G_odd)

    multigraph = nx.MultiGraph()
    multigraph.add_edges_from(T.edges(data=True))
    multigraph.add_edges_from(((u, v, graph[u][v]) for u, v in matching))

    euler_circuit = list(nx.eulerian_circuit(multigraph))

    visited = set()
    path = []
    for u, v in euler_circuit:
        if u not in visited:
            path.append(u)
            visited.add(u)
    path.append(path[0])
    return path

def run_christofides(edges, graph_id, print_details=True):
    G = nx.Graph()
    G.add_edges_from(edges)
    if print_details:
        print(f"\n--- Christofides Grafo {graph_id} ---")

    nodes = list(G.nodes)
    complete_graph = nx.complete_graph(nodes)
    for u, v in complete_graph.edges:
        if G.has_edge(u, v):
            complete_graph[u][v]['weight'] = 1
        else:
            complete_graph[u][v]['weight'] = float('inf')

    start = time.time()
    try:
        path = christofides_tsp(complete_graph)
    except Exception as e:
        if print_details:
            print("Erro ao aplicar Christofides:", e)
        return None
    end = time.time()

    if print_details:
        print("Ciclo Hamiltoniano aproximado:", path)
        print("Caminho formatado:", format_path(path))
        draw_graph(G, graph_id, "christofides", "Christofides", path, edge_color='red')

    return end - start

# --- Algoritmo Held-Karp ---
def held_karp(tsp_matrix):
    n = len(tsp_matrix)
    dp = {}
    parent = {}

    for k in range(1, n):
        dp[(1 << k, k)] = tsp_matrix[0][k]
        parent[(1 << k, k)] = 0

    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            for k in subset:
                prev_bits = bits & ~(1 << k)
                min_dist = float('inf')
                min_prev = -1
                for m in subset:
                    if m == k:
                        continue
                    prev = dp.get((prev_bits, m), float('inf')) + tsp_matrix[m][k]
                    if prev < min_dist:
                        min_dist = prev
                        min_prev = m
                dp[(bits, k)] = min_dist
                parent[(bits, k)] = min_prev

    bits = (1 << n) - 2
    min_cost = float('inf')
    last = -1
    for k in range(1, n):
        cost = dp.get((bits, k), float('inf')) + tsp_matrix[k][0]
        if cost < min_cost:
            min_cost = cost
            last = k

    if min_cost == float('inf'):
        return False, None

    path = [0]
    bits = (1 << n) - 2
    while last != 0:
        path.append(last)
        prev = parent[(bits, last)]
        bits &= ~(1 << last)
        last = prev
    path.append(0)
    path = path[::-1]
    return True, path

def run_heldkarp(edges, graph_id, print_details=True):
    G = nx.Graph()
    G.add_edges_from(edges)
    if print_details:
        print(f"\n--- Held-Karp Grafo {graph_id} ---")

    tsp_matrix, nodes = graph_to_tsp_matrix(G)

    start = time.time()
    is_ham, path = held_karp(tsp_matrix)
    end = time.time()

    if print_details:
        print("É Hamiltoniano?", is_ham)
        if is_ham:
            print("Ciclo Hamiltoniano:", [nodes[i] for i in path])
            print("Caminho formatado:", format_path([nodes[i] for i in path]))
        print("\nMatriz TSP:")
        print("   ", "  ".join(nodes))
        for i, row in enumerate(tsp_matrix):
            row_str = "  ".join(f"{val if val != float('inf') else '-':>3}" for val in row)
            print(f"{nodes[i]}  {row_str}")
        draw_graph(G, graph_id, "held_karp", "Held-Karp", [nodes[i] for i in path] if is_ham else None, edge_color='red')

    return end - start

# --- Algoritmo Lin-Kernighan ---
def graph_to_tsp_matrix_lk(graph):
    nodes = list(graph.nodes)
    n = len(nodes)
    tsp_matrix = np.zeros((n, n))
    lengths = dict(nx.all_pairs_shortest_path_length(graph))

    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i == j:
                tsp_matrix[i][j] = 0
            else:
                dist = lengths[u].get(v, float('inf'))
                tsp_matrix[i][j] = dist if dist != float('inf') else 1000

    return tsp_matrix, nodes

def run_linkernighan(edges, graph_id, print_details=True):
    G = nx.Graph()
    G.add_edges_from(edges)
    if print_details:
        print(f"\n--- Lin-Kernighan Grafo {graph_id} ---")

    tsp_matrix, nodes = graph_to_tsp_matrix_lk(G)

    start = time.time()
    try:
        permutation, cost = solve_tsp_local_search(tsp_matrix)
    except Exception as e:
        if print_details:
            print("Erro ao aplicar Lin-Kernighan:", e)
        return None
    end = time.time()

    if print_details:
        cycle = [nodes[i] for i in permutation] + [nodes[permutation[0]]]
        print("Ciclo Hamiltoniano aproximado:", cycle)
        print("Caminho formatado:", " → ".join(cycle))
        print(f"Custo total (número de arestas): {cost}")
        draw_graph(G, graph_id, "lin_kernighan", "Lin-Kernighan", cycle, edge_color='green')

    return end - start

# --- Grafos para teste ---
grafos = [
    [('A','B'), ('B','C'), ('C','D'), ('D','A')],
    [('A','B'), ('B','C'), ('C','D')],
    [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'E')],
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'A')],
    [('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E')],
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'A')],
    [('A', 'B'), ('B', 'C'), ('C', 'E'), ('E', 'D')],
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('A', 'C')],
    [('A', 'B'), ('B', 'C'), ('C', 'A'), ('C', 'D'), ('D', 'E')],
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'A'), ('B', 'E'), ('C', 'F')]
]

# --- Menu principal ---
def menu():
    while True:
        print("\nEscolha o algoritmo:")
        print("1 - Brute Force")
        print("2 - Christofides")
        print("3 - Held-Karp")
        print("4 - Lin-Kernighan")
        print("5 - Comparar tempos")
        print("0 - Sair")

        choice = input("Opção: ").strip()
        if choice == '0':
            break

        if choice in ['1', '2', '3', '4', '5']:
            if choice == '1':
                for i, edges in enumerate(grafos, 1):
                    run_bruteforce(edges, i)

            elif choice == '2':
                for i, edges in enumerate(grafos, 1):
                    run_christofides(edges, i)

            elif choice == '3':
                for i, edges in enumerate(grafos, 1):
                    run_heldkarp(edges, i)

            elif choice == '4':
                for i, edges in enumerate(grafos, 1):
                    run_linkernighan(edges, i)

            elif choice == '5':
                print("\nComparando tempos dos algoritmos (sem prints intermediários)...")
                total_times = {
                    'Brute Force': 0,
                    'Christofides': 0,
                    'Held-Karp': 0,
                    'Lin-Kernighan': 0
                }

                for i, edges in enumerate(grafos, 1):
                    print(f"\nGrafo {i}:")
                    t1 = run_bruteforce(edges, i, print_details=False)
                    print(f"Brute Force: {t1:.4f} s")
                    total_times['Brute Force'] += t1

                    t2 = run_christofides(edges, i, print_details=False)
                    print(f"Christofides: {t2:.4f} s")
                    total_times['Christofides'] += t2

                    t3 = run_heldkarp(edges, i, print_details=False)
                    print(f"Held-Karp: {t3:.4f} s")
                    total_times['Held-Karp'] += t3

                    t4 = run_linkernighan(edges, i, print_details=False)
                    print(f"Lin-Kernighan: {t4:.4f} s")
                    total_times['Lin-Kernighan'] += t4

                print("\nTempo total de processamento para todos os grafos:")
                for alg, t in total_times.items():
                    print(f"{alg}: {t:.4f} s")

        else:
            print("Opção inválida!")


if __name__ == "__main__":
    menu()
