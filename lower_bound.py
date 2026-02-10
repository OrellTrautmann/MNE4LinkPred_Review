import numpy as np
import matplotlib.pyplot as plt 

def hoeffding_lower_bound(sample_ratio: int, num_edges: int, node_num: int):
    return 1 - np.exp((-2) * (sample_ratio * num_edges) * (1/2 - (num_edges / (node_num * (node_num - 1) - num_edges))) ** 2 )

def serfling_lower_bound(sample_ratio: int, num_edges: int, node_num: int):
    return 1 - np.exp((-2) * (sample_ratio * num_edges) * (1/2 - (num_edges / (node_num * (node_num - 1) - num_edges))) ** 2 * (node_num*(node_num-1)/(node_num*(node_num-1)-sample_ratio*num_edges+1)))

num_node = 100
num_edges = np.arange(num_node - 1, np.floor(num_node*(num_node-1)/3), num_node)
plt.plot(num_edges, hoeffding_lower_bound(0.2, num_edges, num_node), '-b')
plt.plot(num_edges, serfling_lower_bound(0.2, num_edges, num_node), '-g')
plt.plot(num_edges, [0.5]*len(num_edges), '-r')
plt.yscale('linear')
plt.xscale('linear')
plt.xlim((0, num_node*(num_node-1)/3))
plt.ylim((0,1))
plt.show()

