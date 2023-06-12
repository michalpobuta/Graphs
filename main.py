from lab1 import *
from lab2 import *
from lab4 import *
from lab5 import *
from lab6 import *


def lab6():
    # test algorytmu PageRank
    G = nx.gnp_random_graph(10, 0.5, directed=True)
    print("Random walk PageRank:", random_walk_pagerank(G))
    print("Matrix iter PageRank:", matrix_iter_pagerank(G))

    # rysowanie grafu dla lepszego zrozumienia
    nx.draw(G, with_labels=True)
    plt.show()

    # test algorytmu wyszukiwania najkrótszej drogi
    G = nx.complete_graph(10)
    for (u, v, w) in G.edges(data=True):
        w['weight'] = np.random.rand()

    if G.number_of_nodes() > 0 and G.number_of_nodes() > 1:
        print("Shortest route:", simulated_annealing(G))
    else:
        print("Graph has no nodes.")

    # rysowanie grafu dla lepszego zrozumienia
    pos = nx.spring_layout(G)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    nx.draw(G, pos, with_labels=True)
    plt.show()


def lab5():
   # Generowanie losowej sieci przepływowej
    N = 3  # liczba pośrednich warstw
    G = create_flow_network(N)
    source, sink = 1, len(G.nodes())  # numeracja węzłów zaczyna się od 1
    draw_graph(nx.to_numpy_array(G), 1)

    # Obliczanie maksymalnego przepływu
    max_flow = ford_fulkerson(G, source, sink)
    print("Maksymalny przepływ: ", max_flow)

def lab4():
    n = 4  # liczba wierzchołków
    p = 0.5  # prawdopodobieństwo krawędzi
    weight_range = [-5, 10]

    # Generowanie losowego digrafu i zapisanie do pliku csv
    G = generate_digraph(n, p)
    save_data(nx.to_numpy_array(G), "random_digraph")
    draw_graph(nx.to_numpy_array(G), 1)

    # Wczytanie grafu z pliku csv i znalezienie silnie spójnych składowych
    data = load_data(r"data\random_digraph.csv")
    G = nx.from_numpy_array(data)
    components = kosaraju(G)
    print("Silnie spójne składowe: ", components)

   # Generowanie losowego ważonego digrafu
    G_weighted = generate_weighted_digraph(n, p, weight_range)
    save_weighted_data(G_weighted, "random_weighted_digraph")
    draw_graph(nx.to_numpy_array(G_weighted), 1)

    # Wczytanie ważonego grafu z pliku csv i znalezienie najkrótszych ścieżek
    G_weighted_from_file = load_weighted_data("random_weighted_digraph")
    distances = bellman_ford(G_weighted_from_file, 0)
    print("Najkrótsze ścieżki od wierzchołka 0: ", distances)

    # Obliczenie odległości między wszystkimi parami wierzchołków
    all_pairs_distances = johnson(G_weighted_from_file)
    print("Odległości między wszystkimi parami wierzchołków: ", all_pairs_distances)



def main():
    lab6()

if __name__ == "__main__":
    main()




