import os
import sys


# zadanie 2

def algorithm_dijkstra(adjacency_list, weights, num_vertices, start_vertex, verbose=False):
    distances, predecessors = initialize(distances=[float('inf')] * num_vertices, start_vertex=start_vertex)
    visited = []
    while len(visited) != num_vertices:
        unvisited = [vertex for vertex in range(num_vertices) if vertex not in visited]
        current = min(unvisited, key=lambda vertex: distances[vertex])
        visited.append(current)
        neighbors = [vertex for vertex in adjacency_list[current] if vertex not in visited]
        for neighbor in neighbors:
            relax(current, neighbor, weights, distances, predecessors)
    if verbose:
        print(f"START: {start_vertex}")
        for vertex in range(num_vertices):
            print(f"Distance to {vertex} = {distances[vertex]} ==>", end=" ")
            print("[", end="")
            path = [vertex]
            previous = predecessors[vertex]
            while previous != "NIL":
                path.append(previous)
                previous = predecessors[previous]
            path = path[::-1]
            for value in path:
                if value != path[-1]:
                    print(f"{value} - ", end="")
                else:
                    print(f"{value}]")
    return distances


def initialize(distances, start_vertex):
    distances[start_vertex] = 0
    predecessors = ["NIL"] * len(distances)
    return distances, predecessors


def relax(current, neighbor, weights, distances, predecessors):
    new_distance = distances[current] + weights[(current, neighbor)]
    if new_distance < distances[neighbor]:
        distances[neighbor] = new_distance
        predecessors[neighbor] = current




# zadanie 3

def dijkstra(start_vertex, adjacency_list, weights, num_vertices):
    distances, predecessors = initialize(distances=[float('inf')] * num_vertices, start_vertex=start_vertex)
    visited = []
    while len(visited) != num_vertices:
        unvisited = [vertex for vertex in range(num_vertices) if vertex not in visited]
        current = min(unvisited, key=lambda vertex: distances[vertex])
        visited.append(current)
        neighbors = [vertex for vertex in adjacency_list[current] if vertex not in visited]
        for neighbor in neighbors:
            relax(current, neighbor, weights, distances, predecessors)
    return distances



def distance_matrix(adjacency_list, weights):
    num_vertices = len(adjacency_list)
    distances = []
    for vertex in range(num_vertices):
        distances_row = dijkstra(vertex, adjacency_list, weights, num_vertices)
        distances.append(distances_row)
    for i, row in enumerate(distances):
        print(f"Distances from vertex {i}:")
        for j, dist in enumerate(row):
            print(f"  To vertex {j}: {dist}")
        print()

# zadanie 4

def calculate_graph_center(adjacency_list, weights):
    num_vertices = len(adjacency_list)
    distances = []
    for vertex in range(num_vertices):
        distances_row = dijkstra(vertex, adjacency_list, weights, num_vertices)
        distances.append(distances_row)
    center = distances.index(min(distances, key=lambda x: sum(x)))
    center_sum = sum(distances[center])
    print(f"The center of the graph is vertex {center} with a total distance of {center_sum}.")
    minimax = distances.index(min(distances, key=lambda x: max(x)))
    minimax_distance = max(distances[minimax])
    print(f"The minimax center of the graph is vertex {minimax} with a distance of {minimax_distance} from the farthest vertex.")



# zadanie 5


def get_minimum_spanning_tree(weights):
    edges_to_weights = {}
    for j in range(len(weights)):
        for i in range(len(weights[j])):
            if (not (j,i) in edges_to_weights) and j != i and weights[i][j] != 0:
                edges_to_weights[(i,j)] = weights[i][j]
    sorted_edges_to_weights = {k: v for k, v in sorted(edges_to_weights.items(), key=lambda item: item[1])}
    
    sets = {v: set([v]) for edge in sorted_edges_to_weights for v in edge}
    mst = []
    for edge in sorted_edges_to_weights:
        if sets[edge[0]] != sets[edge[1]]:
            mst.append(edge)
            set1 = sets[edge[0]]
            set2 = sets[edge[1]]
            union = set1.union(set2)
            for v in union:
                sets[v] = union
                
    return mst
