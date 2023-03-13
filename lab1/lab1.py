import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random


def load_data(data_source):
    # wczytywanie danych
    data = pd.read_csv(data_source, sep=";", header=None)
    data = np.array(data)
    return data


def save_data(data, file_name):
    with open(f"{file_name}.csv", "a+") as f:
        np.savetxt(fr"dane\{file_name}.csv", data, delimiter=";", fmt='%i')


def change_encoding(data, data_module, expected_module):
    # 1 -> macierz sąsiedztwa
    # 2 -> macierz incydencji
    # 3 -> lista sąsiedztwa
    if data_module == 1 and expected_module == 2:
        rows = data.shape[0]
        cols = data.shape[1]
        connections = int(data.sum() / 2)
        output_data = np.zeros((rows,connections), dtype="int")
        data_triu = np.triu(data, k = 1)

        edge = 0
        for row in range(rows):
            for col in range(row+1, cols):
                if data_triu[row, col] == 1:
                    output_data[row, edge] = 1
                    output_data[col, edge] = 1
                    edge += 1

        return output_data.astype(int)

    elif data_module == 1 and expected_module == 3:
        rows = data.shape[0]
        cols = data.shape[1]
        output_data = np.empty((rows, rows - 1)) * np.nan
        for row in range(rows):
            output_col = 0
            for col in range(cols):
                if data[row, col] == 1:
                    output_data[row, output_col] = col
                    output_col += 1

        return output_data

    elif data_module == 2 and expected_module == 1:
        rows = data.shape[0]
        cols = data.shape[1]
        output_data = np.zeros((rows, rows), dtype="int")

        for col in range(cols):
            temp_row = None
            temp_col = None
            for row in range(rows):
                if data[row, col] == 1:
                    if temp_row == None:
                        temp_row = row
                    else:
                        temp_col = row
            output_data[temp_row, temp_col] = 1
            output_data[temp_col, temp_row] = 1

        return output_data

    elif data_module == 2 and expected_module == 3:
        temp_data = change_encoding(data, 2, 1)
        return change_encoding(temp_data, 1, 3)

    elif data_module == 3 and expected_module == 1:
        rows = data.shape[0]
        cols = data.shape[1]
        output_data = np.zeros((rows, rows))

        for row in range(rows):
            for col in range(cols):
                temp_value = data[row, col]
                if not np.isnan(temp_value):
                    output_data[row, int(temp_value)] = 1
                    output_data[int(temp_value), row] = 1

        return output_data.astype(int)

    elif data_module == 3 and expected_module == 2:
        temp_data = change_encoding(data, 3, 1)
        return change_encoding(temp_data, 1, 2)


def draw_graph(data, module):
    if module == 1:
        pass
    elif module == 2:
        data = change_encoding(data, 2, 1)
    elif module == 3:
        data = change_encoding(data, 3, 1)

    graph = nx.from_numpy_array(data)
    nx.draw_circular(graph, with_labels = True)
    plt.axis('equal')
    plt.show()


def make_random_graph(n, l=None, p=None):
    output_graph = np.zeros((n, n))
    if l:
        if l > n * (n - 1) / 2:
            print("Liczba krawędzi jest większa od liczby możliwych krawędzi")

        edge_number = 0

        while edge_number < l:
            x1, x2 = random.sample(range(n), 2)
            if output_graph[x1, x2] == 0 and x1 != x2:
                output_graph[x1, x2] = 1
                output_graph[x2, x1] = 1
                edge_number = int(output_graph.sum() / 2)

        return output_graph.astype(int)

    elif p:
        for row in range(n):
            for col in range(row + 1, n):
                if random.random() < p:
                    output_graph[row, col] = 1
                    output_graph[col, row] = 1

        return output_graph.astype(int)


if __name__ == "__main__":
    # data = load_data(r"dane\test1.csv")
    # draw_graph(data, 3)
    data = make_random_graph(50, p=0.1)
    draw_graph(data, 1)

