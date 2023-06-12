import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_flow_network(N):
    # Tworzenie sieci przepływowej z N warstwami
    G = nx.DiGraph()
    layers = [0]

    # Tworzenie wierzchołków dla każdej warstwy
    for i in range(1, N+1):
        num_nodes = random.randint(2, N)
        for _ in range(num_nodes):
            G.add_node(len(layers), layer=i)
            layers.append(i)

    # Dodanie ujścia
    G.add_node(len(layers), layer=N+1)
    layers.append(N+1)

    # Łączenie wierzchołków między warstwami
    for node in G.nodes():
        if G.nodes[node]['layer'] < N+1:
            next_layer_nodes = [n for n in G.nodes() if G.nodes[n]['layer'] == G.nodes[node]['layer'] + 1]
            for next_node in next_layer_nodes:
                G.add_edge(node, next_node, capacity=random.randint(1, 10))

    # Dodanie dodatkowych łuków
    for _ in range(2*N):
        possible_edges = [(u, v) for u in G.nodes() for v in G.nodes() if u != v and G.nodes[u]['layer'] < G.nodes[v]['layer'] and not G.has_edge(u, v)]
        if possible_edges:
            u, v = random.choice(possible_edges)
            G.add_edge(u, v, capacity=random.randint(1, 10))

    return G

def bfs_path(G, source, sink):
    # Wyszukiwanie ścieżki powiększającej za pomocą BFS
    queue = [(source, [], float('inf'))]
    while queue:
        node, path, min_cap = queue.pop(0)
        path = path + [node]
        for neighbor in set(G[node].keys()) - set(path):
            edge_cap = G.edges[node, neighbor]['capacity']
            if edge_cap > 0:
                if neighbor == sink:
                    return path + [neighbor]
                queue.append((neighbor, path, min(edge_cap, min_cap)))
    return None


def ford_fulkerson(G, source, sink):
    # Algorytm Forda-Fulkersona
    max_flow = 0
    path = bfs_path(G, source, sink)
    while path:
        flow = min(G.edges[path[i-1], path[i]]['capacity'] for i in range(1, len(path)))
        for i in range(1, len(path)):
            u, v = path[i-1], path[i]
            G.edges[u, v]['capacity'] -= flow
            if not G.has_edge(v, u):
                G.add_edge(v, u, capacity=0)
            G.edges[v, u]['capacity'] += flow
        max_flow += flow
        path = bfs_path(G, source, sink)
    return max_flow
