import numpy as np
import networkx as nx
import seaborn as sn
from tqdm import tqdm

SEED = 1234
GRAPH_SIZE = 100

# number of sampled reciprocal edges simulation using uniform sampling procedure

def uniform_neg_sampling(edgetuple_list: list, node_list: list, sample_size):
    neg_edgetuple_list = list()
    for _ in range(sample_size):
        edge = (np.random.randint(1,max(node_list)+1), np.random.randint(1,max(node_list)+1))
        while edge in edgetuple_list or edge in neg_edgetuple_list:
            edge = (np.random.randint(1,max(node_list)+1), np.random.randint(1,max(node_list)+1))
        
        neg_edgetuple_list.append(edge)

    return neg_edgetuple_list

# number of sampled reciprocal edges simulation using degree-based sampling procedure




def find_reciprocals(edgetuple_list: list, neg_edgetuple_list: list):
    count = 0
    for (source, target) in neg_edgetuple_list:
        if (target, source) in edgetuple_list:
            count += 1
    return count


if __name__ == "__main__":
    np.random.seed(SEED)

    for density in tqdm([x / 10.0 for x in range(1, 7, 1)]):

        list_num_reciprocals = list()

        for _ in range(10):
            digraph = nx.erdos_renyi_graph(GRAPH_SIZE, density, seed=np.random.randint(0,1e8), directed=True)

            neg_ratio = .2
            edgetuple_list = digraph.edges
            node_list = digraph.nodes
            sample_size = round(neg_ratio * len(edgetuple_list))

            neg_edgetuple_list = uniform_neg_sampling(edgetuple_list, node_list, sample_size)

            num_reciprocals = find_reciprocals(edgetuple_list, neg_edgetuple_list)

            list_num_reciprocals.append(num_reciprocals / sample_size)

        print(list_num_reciprocals)