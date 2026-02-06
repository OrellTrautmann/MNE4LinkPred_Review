import numpy as np
import networkx as nx
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import (seed_everything,
                   uniform_neg_sampling,
                   degree_neg_sampling)

def uniform_expected_neg_samp(network_denstity: float, neg_ratio = .2):
    sample_size = network_denstity * GRAPH_SIZE * (GRAPH_SIZE - 1) * neg_ratio
    return sample_size / (1 / network_denstity + 1)

def find_reciprocals(edgetuple_list: list, neg_edgetuple_list: list):
    count = 0
    for (source, target) in neg_edgetuple_list:
        if (target, source) in edgetuple_list:
            count += 1
    return count

def plot_ExpectionsVSSampled(neg_ratio: float, graph_size: int, seed: int = 1234):
    seed_everything(seed)
    # Simulation of the number of reciprocal edges sampled plotted together
    # with the expected value curve w.r.t. the density.

    densities = np.linspace(1e-3, 2e-1, 10**3)

    
    fig, ax = plt.subplots() 

    ax.plot(densities, uniform_expected_neg_samp(densities, neg_ratio), 'r-', label='Expected number of reciprocals in sample')
    ax.set_xlabel("densities")
    ax.set_ylabel("Number of negative reciprocal samples")
    ax.set_xscale("log")
    ax.set_yscale("log")

    list_mean_num_rec = list()
    list_densities = list()
    list_errors_upper = list()
    list_errors_lower = list()

    degree_list_mean_num_rec = list()
    degree_list_errors_upper = list()
    degree_list_errors_lower = list()

    for density in tqdm([x / 1e3 for x in range(10, 191, 20)]):

        list_num_reciprocals = list()
        degree_list_num_reciprocals = list()

        for _ in range(10):
            digraph = nx.fast_gnp_random_graph(graph_size, density, seed=np.random.randint(0,1e8), directed=True)

            edgetuple_list = digraph.edges
            node_list = digraph.nodes
            sample_size = round(neg_ratio * len(edgetuple_list))

            neg_edgetuple_list = uniform_neg_sampling(edgetuple_list, node_list, sample_size)
            num_reciprocals = find_reciprocals(edgetuple_list, neg_edgetuple_list)
            list_num_reciprocals.append(num_reciprocals)

            degree_neg_edgetuple_list = degree_neg_sampling(edgetuple_list, node_list, sample_size)
            degree_num_reciprocals = find_reciprocals(edgetuple_list, degree_neg_edgetuple_list)
            degree_list_num_reciprocals.append(degree_num_reciprocals)

        array_num_reciprocals = np.array(list_num_reciprocals)
        list_mean_num_rec.append(np.mean(array_num_reciprocals))
        list_errors_lower.append(np.abs(np.min(array_num_reciprocals - list_mean_num_rec[-1])))
        list_errors_upper.append(np.max(array_num_reciprocals - list_mean_num_rec[-1]))
        list_densities.append(density)

        degree_array_num_reciprocals = np.array(degree_list_num_reciprocals)
        degree_list_mean_num_rec.append(np.mean(degree_array_num_reciprocals))
        degree_list_errors_lower.append(np.abs(np.min(degree_array_num_reciprocals - degree_list_mean_num_rec[-1])))
        degree_list_errors_upper.append(np.max(degree_array_num_reciprocals - degree_list_mean_num_rec[-1]))
    

    ax.errorbar(x=np.array(list_densities), y=np.array(list_mean_num_rec), 
                yerr=[np.array(list_errors_lower), np.array(list_errors_lower)], 
                fmt='o', capsize=2, capthick=1, markersize=3, label='uniform sampled reciprocals',
                markerfacecolor='black', ecolor='black')



    ax.errorbar(x=np.array(list_densities), y=np.array(degree_list_mean_num_rec), 
                yerr=[np.array(degree_list_errors_lower), np.array(degree_list_errors_lower)], 
                fmt='o', capsize=2, capthick=1, markersize=3, label='degree sampled reciprocals',
                markerfacecolor='blue', ecolor='blue')

    ax.legend()
    fig.savefig("neg_sampling_plot.png")

if __name__ == "__main__":
    SEED = 1234
    GRAPH_SIZE = 1000
    print("fin")
    #plot_ExpectionsVSSampled(neg_ratio = .2, graph_size=GRAPH_SIZE, seed=SEED)
    
    
