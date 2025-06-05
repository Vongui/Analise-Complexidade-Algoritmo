import networkx as nx
import itertools
import numpy as np
import matplotlib.pyplot as plt

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

def format_path(path):
    return " → ".join(path)

def draw_graph(graph, graph_id, highlight_path=None):
    plt.figure(figsize=(6, 6))
    plt.title(f"Grafo {graph_id}")
    pos = nx.spring_layout(graph, seed=42)

    nx.draw(graph, pos, with_labels=True, node_color='skyblue',
            node_size=500, edge_color='gray', font_weight='bold')

    if highlight_path:
        edges = list(zip(highlight_path, highlight_path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='red', width=2)

    plt.savefig(f"grafo_christofides_{graph_id}.png")
    plt.close()

def process_graph_christofides(edges, graph_id):
    G = nx.Graph()
    G.add_edges_from(edges)

    print(f"\n--- Grafo {graph_id} (Christofides) ---")

    nodes = list(G.nodes)
    complete_graph = nx.complete_graph(nodes)
    for u, v in complete_graph.edges:
        if G.has_edge(u, v):
            complete_graph[u][v]['weight'] = 1
        else:
            complete_graph[u][v]['weight'] = float('inf')

    try:
        path = christofides_tsp(complete_graph)
        print("Ciclo Hamiltoniano aproximado:", path)
        print("Caminho formatado:", format_path(path))
        draw_graph(G, graph_id, highlight_path=path)
    except Exception as e:
        print("Erro ao aplicar algoritmo de Christofides:", e)

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
        process_graph_christofides(edges, i)

if __name__ == "__main__":
    main()
