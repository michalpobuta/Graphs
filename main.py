from lab1 import *
from lab2 import *

#LAB1
# data = load_data(r"data\test1.csv")
# draw_graph(data, 3)
#
# data = make_random_graph(10, p=0.87)
# draw_graph(data, 1)
# save_data(data, "output")

#LAB2
graph = construct_graph([2, 2, 3, 2, 1, 4, 2, 2, 2, 6,0, 0, 0])

draw_graph(graph, 1)
graph = randomize_graph(graph, 10)
draw_graph(graph, 1)

all_compononts = all_connected_components(graph)
draw_graph(graph, 1, groups=all_compononts)



