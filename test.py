import pandas as pd

df = pd.read_csv("Datasets/SacchCere/sacchcere_genetic_multiplex.edges.txt", header=None, sep=" ")

print(df.head(5))

node_list = list(set(df[1]).union(df[2]))
print(1)
max_node = max(node_list)
print(2)

print(f"list of missing nodes: {list(set(node_list).difference(set(range(max_node))))}")

print(node_list[0:10], list(range(max_node))[0:10])
print(node_list[-10:], list(range(max_node))[-10:])
print(len(node_list), len(list(range(max_node))))