from lab1 import *


def is_graphic_sequence(seq):
    seq = np.sort(seq)[::-1]
    while seq[0] > 0:
        if seq[0] >= len(seq):
            return False
        seq[1:seq[0] + 1] -= 1
        if np.any(seq[1:seq[0] + 1] < 0):
            return False
        seq = np.sort(seq[1:])[::-1]
    return True


def construct_graph(seq):
    if is_graphic_sequence(seq):
        n = len(seq)
        graph = np.zeros((n, n), dtype=int)
        seq = np.sort(seq)[::-1]
        for i in range(n):
            for j in range(i + 1, n):
                if seq[i] > 0 and seq[j] > 0:
                    graph[i, j] = 1
                    graph[j, i] = 1
                    seq[i] -= 1
                    seq[j] -= 1
        return graph
    else:
        return None


# zadanie 2 do naprawienia


def random_edge_swap(graph):
    n = graph.shape[0]
    # losowo wybieramy 2 pary krawędzi
    a, b, c, d = random.sample(range(n), 4)
    while b == a or c == d or graph[a, b] == 0 or graph[c, d] == 0 or graph[a, d] == 1 or graph[b, c] == 1:
        a, b, c, d = random.sample(range(n), 4)
    # zamieniamy krawędzie
    graph[a, b] = 0
    graph[b, a] = 0
    graph[c, d] = 0
    graph[d, c] = 0
    graph[a, d] = 1
    graph[d, a] = 1
    graph[b, c] = 1
    graph[c, b] = 1
    return graph


# funkcja randomizująca graf
def randomize_graph(graph, num_iterations):
    for i in range(num_iterations):
        graph = random_edge_swap(graph)
    return graph




# zadanie 3
# def dfs(graph, visited, v, component):
#     visited[v] = True
#     component.append(v)
#     for i in range(len(graph)):
#         if graph[v][i] == 1 and not visited[i]:
#             dfs(graph, visited, i, component)
#
#
#
# def largest_connected_component(graph):
#     n = len(graph)
#     visited = [False] * n
#     largest_component = []
#     for i in range(n):
#         if not visited[i]:
#             component = []
#             dfs(graph, visited, i, component)
#             if len(component) > len(largest_component):
#                 largest_component = component
#     return largest_component


def dfs(graph, visited, v, component):
    visited[v] = True
    component.append(v)
    for i in range(len(graph)):
        if graph[v][i] == 1 and not visited[i]:
            dfs(graph, visited, i, component)


def longest_subarray(arr):
    max_len = 0
    max_subarr = []
    for subarr in arr:
        if len(subarr) > max_len:
            max_len = len(subarr)
            max_subarr = subarr
    return max_subarr



def all_connected_components(graph):
    n = len(graph)
    visited = [False] * n
    all_components = []
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(graph, visited, i, component)
            all_components.append(component)
    print("Największa składowa grafu: " + str(longest_subarray(all_components)))
    return all_components