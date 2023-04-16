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


# def random_edge_swap(graph):
#     n = graph.shape[0]
#     # losowo wybieramy 2 pary krawędzi
#     a, b, c, d = random.sample(range(n), 4)
#     while b == a or c == d or graph[a, b] == 0 or graph[c, d] == 0 or graph[a, d] == 1 or graph[b, c] == 1:
#         a, b, c, d = random.sample(range(n), 4)
#     # zamieniamy krawędzie
#     graph[a, b] = 0
#     graph[b, a] = 0
#     graph[c, d] = 0
#     graph[d, c] = 0
#     graph[a, d] = 1
#     graph[d, a] = 1
#     graph[b, c] = 1
#     graph[c, b] = 1
#     return graph


# # funkcja randomizująca graf
# def randomize_graph(graph, num_iterations):
#     for i in range(num_iterations):
#         graph = random_edge_swap(graph)
#     return graph


def randomize_edges(neighbourList, number, verbose=False):
    edges = []
    numberOfEdges = 0
    for index, neighbours in enumerate(neighbourList):
        edges.extend((index, neighbour) for neighbour in neighbours)
        numberOfEdges += len(neighbours)
    if numberOfEdges == 0:
        return edges
    i = 0
    while i < number:
        randAB = np.random.randint(0, numberOfEdges)
        randCD = randAB
        while randCD == randAB:
            randCD = np.random.randint(0, numberOfEdges)

        if verbose:
            print(f"{edges[randAB]}, {edges[randCD]} => ", end="")

        AB = list(edges[randAB])
        CD = list(edges[randCD])
        AB[0], AB[1], CD[0], CD[1] = AB[0], CD[1], AB[1], CD[0]

        reversed = AB[::-1]
        if (
            tuple(AB) not in edges
            and tuple(reversed) not in edges
            and tuple(CD) not in edges
            and tuple(CD[::-1]) not in edges
            and AB[0] != AB[1]
            and CD[0] != CD[1]
            and AB != CD
        ):
            edges[randAB], edges[randCD] = tuple(AB), tuple(CD)
            if verbose:
                print(f"{edges[randAB]}, {edges[randCD]}")
        else:
            if verbose:
                print("Skip..")
            i -= 1
        i += 1

    return edges



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



# zad 4
def generate_graph_NL(n, l):
    if l < 0 or l > n * (n - 1) / 2:
        raise ValueError("Wrong randomization arguments")

    edges = []

    for i in range(l):
        while True:
            rand1 = np.random.randint(1, n + 1)
            rand2 = rand1

            while rand1 == rand2:
                rand2 = np.random.randint(1, n + 1)

            edge = [rand1, rand2]
            edge.sort()

            if edge not in edges:
                edges.append(edge)
                break

    maxList = map(max, edges)

    nodeNumber = max(maxList)
    neighbourList = []

    for _ in range(nodeNumber):
        neighbourList.append([])

    for edge in edges:
        neighbourList[edge[0] - 1].append(edge[1])
        neighbourList[edge[1] - 1].append(edge[0])

    return neighbourList

def calculateIntegrityArrayV3(sequence, neighbourList):
    while [] in neighbourList:
        emptyArrayIndex = neighbourList.index([])
        neighbourList.pop(emptyArrayIndex)
        for i in range(len(neighbourList)):
            for j in range(len(neighbourList[i])):
                if neighbourList[i][j] > emptyArrayIndex + 1:
                    neighbourList[i][j] = neighbourList[i][j] - 1
        sequence = [len(neighbourList[i]) for i in range(len(neighbourList))]

    numberOfNodes = len(sequence)
    comp = []
    integrityNumber = 0

    for i in range(numberOfNodes):
        comp.append(-1)

    def componentsV3(integrityNumber, index, neighbourList, comp):
        neighbours = neighbourList[index]

        for i in neighbours:
            if comp[i - 1] == -1:
                comp[i - 1] = integrityNumber
                componentsV3(integrityNumber, i - 1, neighbourList, comp)

    for i in range(numberOfNodes):
        if comp[i] == -1:
            integrityNumber += 1
            comp[i] = integrityNumber
            componentsV3(integrityNumber, i, neighbourList, comp)

    return comp


def generateEulerGraph(n, l):
    while True:
        neighbourList = generate_graph_NL(n, l)
        sequence = [len(neighbourList[i]) for i in range(len(neighbourList))]

        if max(calculateIntegrityArrayV3(sequence, neighbourList)) == 1 and all(
            sequence[i] % 2 == 0 for i in range(len(sequence))
        ):
            return neighbourList


# zad 5
def generateRegularGraphNP(nodes, level, probability=None):
    if level >= nodes or (level % 2 == 1 and not nodes % 2 == 0):
        raise ValueError("Wrong randomization arguments")

    if probability is None:
        probability = level / nodes

    while True:
        edges = []

        for i in range(1, nodes + 1):
            for j in range(i + 1, nodes + 1):
                rand = np.random.rand()

                if rand < probability:
                    edges.append([i, j])

        if edges != []:
            if max(map(max, edges)) == nodes:
                neighbourList = [[] for _ in range(nodes)]

                for edge in edges:
                    neighbourList[edge[0] - 1].append(edge[1])
                    neighbourList[edge[1] - 1].append(edge[0])

                sequence = [len(neighbourList[i]) for i in range(len(neighbourList))]

                if all(sequence[i] == level for i in range(len(sequence))):
                    return neighbourList
                

# draw graphs

def draw_graph_NL_EL(data, inputType="NL"):
    G = nx.Graph()
    if inputType == "NL":
        G.add_nodes_from(range(1, len(data) + 1))
        G.add_edges_from([(index, neighbour) for index, neighbours in enumerate(data) for neighbour in neighbours])
    elif inputType == "EL":
        G.add_edges_from(data)
        G.add_nodes_from(range(1, len(data) + 1))
    pos = nx.circular_layout(G)
    node_labels = {i: i for i in range(1, len(data) + 1)}
    nx.draw(
        G,
        pos=pos,
        node_size=1000,
        node_color="#d9d9ff",
        node_shape="o",
        linewidths=1.0,
        edgecolors="#1414ff",
        edge_color="#000",
        width=2,
        labels=node_labels,
        font_size=16.0,
        font_color="#000",
    )
    plt.show()