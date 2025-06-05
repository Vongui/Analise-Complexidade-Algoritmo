import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from python_tsp.heuristics import solve_tsp_local_search

def graph_to_tsp_matrix(graph):
    nodes = list(graph.nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
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

def draw_graph(graph, graph_id, highlight_path=None):
    plt.figure(figsize=(6, 6))
    plt.title(f"Grafo {graph_id} (Lin-Kernighan)")
    pos = nx.spring_layout(graph, seed=42)

    nx.draw(graph, pos, with_labels=True, node_color='skyblue',
            node_size=500, edge_color='gray', font_weight='bold')

    if highlight_path:
        edges = list(zip(highlight_path, highlight_path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='green', width=2)

    plt.savefig(f"grafo_linkernighan_{graph_id}.png")
    plt.close()

def process_graph_lk(edges, graph_id):
    G = nx.Graph()
    G.add_edges_from(edges)

    print(f"\n--- Grafo {graph_id} (Lin-Kernighan) ---")
    tsp_matrix, nodes = graph_to_tsp_matrix(G)

    try:
        permutation, cost = solve_tsp_local_search(tsp_matrix)

        cycle = [nodes[i] for i in permutation] + [nodes[permutation[0]]]

        print("Ciclo Hamiltoniano aproximado:", cycle)
        print("Caminho formatado:", " → ".join(cycle))
        print(f"Custo total (número de arestas): {cost}")

        draw_graph(G, graph_id, highlight_path=cycle)
    except Exception as e:
        print("Erro ao aplicar Lin-Kernighan:", e)

def main():
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

    for i, edges in enumerate(grafos, 1):
        process_graph_lk(edges, i)

if __name__ == "__main__":
    main()
