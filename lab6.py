import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

def random_walk_pagerank(G, d=0.15, epochs=10000):
    num_nodes = len(G.nodes)
    rank = np.zeros(num_nodes)
    curr_node = np.random.choice(G.nodes)

    for _ in range(epochs):
        rank[curr_node] += 1
        if np.random.rand() < d or len(list(G[curr_node])) == 0:
            curr_node = np.random.choice(G.nodes)
        else:
            curr_node = np.random.choice(list(G[curr_node]))

    return rank / epochs

def matrix_iter_pagerank(G, d=0.15, epochs=100):
    A = nx.to_numpy_array(G)
    num_nodes = len(G.nodes)
    out_degrees = np.sum(A, axis=0)
    P = (1 - d) * A / out_degrees + d / num_nodes

    rank = np.ones(num_nodes) / num_nodes
    for _ in range(epochs):
        rank = rank @ P

    return rank


def total_distance(G, route):
    return sum(G.edges[route[i-1], route[i]]['weight'] for i in range(len(route)))

def two_opt(route, i, j):
    new_route = copy.deepcopy(route)
    new_route[i:j+1] = list(reversed(route[i:j+1]))
    return new_route

def simulated_annealing(G, temp=1000, cooling_rate=0.999, num_iter=1000):
    nodes = list(G.nodes)
    best_route = np.random.permutation(nodes)
    best_distance = total_distance(G, best_route)

    for _ in range(num_iter):
        temp *= cooling_rate
        new_route = two_opt(best_route, np.random.randint(len(nodes)), np.random.randint(len(nodes)))
        new_distance = total_distance(G, new_route)
        if (new_distance < best_distance or
            np.random.rand() < np.exp((best_distance - new_distance) / temp)):
            best_route, best_distance = new_route, new_distance

    return best_route

