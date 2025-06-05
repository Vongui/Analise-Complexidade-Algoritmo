import networkx as nx
import itertools
import numpy as np
import matplotlib.pyplot as plt

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

def format_path(path):
    return " → ".join(path)

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

def draw_graph(graph, cycle=None, graph_id=None):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=700)
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges, width=2)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family="sans-serif")
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    if cycle:
        cycle_edges = list(zip(cycle, cycle[1:] ))
        nx.draw_networkx_edges(graph, pos, edgelist=cycle_edges, edge_color='r', width=3)
    plt.title(f"Grafo {graph_id}")
    plt.axis('off')
    plt.savefig(f'grafo_{graph_id}.png')
    plt.close()

def process_graph(edges, graph_id):
    G = nx.Graph()
    G.add_edges_from(edges)

    print(f"\n--- Grafo {graph_id} ---")
    is_ham, cycle = is_hamiltonian(G)
    print("É Hamiltoniano?", is_ham)
    if is_ham:
        print("Ciclo Hamiltoniano:", cycle)
        print("Caminho formatado:", format_path(cycle))

    tsp_matrix, nodes = graph_to_tsp_matrix(G)
    print("\nMatriz TSP:")
    print("   ", "  ".join(nodes))
    for i, row in enumerate(tsp_matrix):
        row_str = "  ".join(f"{val if val != float('inf') else '∞':>3}" for val in row)
        print(f"{nodes[i]}  {row_str}")

    draw_graph(G, cycle if is_ham else None, graph_id)

def main():
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
        process_graph(edges, i)

if __name__ == "__main__":
    main()
