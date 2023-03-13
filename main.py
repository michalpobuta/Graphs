from lab1 import *

data = load_data(r"dane\test1.csv")
draw_graph(data, 3)

data = make_random_graph(10, p=0.87)
draw_graph(data, 1)
save_data(data, "output")
