from collections import deque

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_digraph(n, p):
    return nx.gnp_random_graph(n, p, directed=True)

def save_graph_to_csv(G, filename):
    adjacency_matrix = nx.to_numpy_array(G)
    np.savetxt(filename, adjacency_matrix, delimiter=",")

def kosaraju(G):
    def dfs(G, v, visited, order):
        visited.add(v)
        for neighbor in G[v]:
            if neighbor not in visited:
                dfs(G, neighbor, visited, order)
        order.appendleft(v)

    def reverse_graph(G):
        R = nx.DiGraph()
        for v in G.nodes():
            for neighbor in G[v]:
                R.add_edge(neighbor, v)
        return R

    order = deque()
    visited = set()
    for v in G.nodes():
        if v not in visited:
            dfs(G, v, visited, order)

    G_rev = reverse_graph(G)

    visited = set()
    components = []
    while order:
        v = order.popleft()
        if v not in visited:
            component = deque()
            dfs(G_rev, v, visited, component)
            components.append(component)

    return components

import random

def generate_weighted_digraph(n, p, weight_range):
    G = generate_digraph(n, p)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = random.randint(*weight_range)
    return G

def bellman_ford(G, source):
    distance = {v: float('inf') for v in G.nodes()}
    distance[source] = 0

    for _ in range(len(G) - 1):
        for (u, v) in G.edges():
            if distance[u] + G.edges[u, v]['weight'] < distance[v]:
                distance[v] = distance[u] + G.edges[u, v]['weight']

    for (u, v) in G.edges():
        if distance[u] + G.edges[u, v]['weight'] < distance[v]:
            raise nx.NetworkXUnbounded("Negative cycle detected")

    return distance

def johnson(G):
    def initialize_single_source(G, source):
        distance = {v: float('inf') for v in G.nodes()}
        distance[source] = 0
        return distance

    def relax(u, v, distance):
        if distance[u] + G[u][v]['weight'] < distance[v]:
            distance[v] = distance[u] + G[u][v]['weight']

    G_new = G.copy()
    G_new.add_node('source')
    for v in G.nodes():
        G_new.add_edge('source', v, weight=0)

    try:
        h = bellman_ford(G_new, 'source')
    except nx.NetworkXUnbounded:
        raise nx.NetworkXUnbounded("Negative cycle detected")

    for (u, v) in G.edges():
        G[u][v]['weight'] += h[u] - h[v]

    distance = {}
    for v in G.nodes():
        D = initialize_single_source(G, v)
        for _ in range(len(G) - 1):
            for (u, v) in G.edges():
                relax(u, v, D)
        for u in G.nodes():
            D[u] += h[u] - h[v]
        distance[v] = D

    return distance
