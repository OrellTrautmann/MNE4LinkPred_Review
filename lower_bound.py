import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

def chvatal_lower_bound(sample_ratio: int, num_edges: int, node_num: int, c: float = 1):
    if c/(c+1) * node_num * (node_num - 1) <= num_edges:
        return 'NA'
    else:
        return str(np.round(1 - np.exp((-2) * (sample_ratio * num_edges) * (c - (num_edges / (node_num * (node_num - 1) - num_edges))) ** 2 ),2))

def serfling_lower_bound(sample_ratio: int, num_edges: int, node_num: int, c: float = 1):
    if c/(c+1) * node_num * (node_num - 1) <= num_edges:
        return 'NA'
    else:
        return str(np.round(1 - np.exp((-2) * (sample_ratio * num_edges) * (c - (num_edges / (node_num * (node_num - 1) - num_edges))) ** 2 * (node_num*(node_num-1)/(node_num*(node_num-1)-sample_ratio*num_edges+1))),2))
    

def density(num_edges: int, node_num: int):
    return str(np.format_float_scientific(num_edges/(node_num * (node_num - 1)), precision=2))


datasets = [
    "Datasets/CKM/CKM-Physicians-Innovation_multiplex.edges.txt",
    "Datasets/SacchCere/sacchcere_genetic_multiplex.edges.txt",
    "Datasets/Twitter/Twitter.edges.txt",
    "Datasets/Vickers/Vickers-Chan-7thGraders_multiplex.edges.txt"
]

results = list()
TEST_RATIO = .2

for file_path in datasets:
    multiplex_df = pd.read_csv(file_path, sep=" ", header=None)
    multiplex_df.columns = ['layer', 'source', 'target', 'weight']

    for layer in multiplex_df['layer'].unique():
        layer_network_df = multiplex_df[multiplex_df['layer'] == layer]
        num_edges = layer_network_df.shape[0]
        num_nodes = len(set(layer_network_df['source']).union(set(layer_network_df['target'])))

        bounds = {'dataset': file_path.split('/')[1], 'layer': int(layer), 'density': density(num_edges, num_nodes)}
        for c in [0.5, 0.1, 0.05, 0.01]:
            bounds.update({f'chvatal c={c}': chvatal_lower_bound(TEST_RATIO, num_edges, num_nodes, c)})
                           #f'serfling c={c}': serfling_lower_bound(TEST_RATIO, num_edges, num_nodes, c)})
        
        results.append(bounds)
        

results_df = pd.DataFrame(results).round(2)  

latex_table = results_df.to_latex(index=False)

print(latex_table)     
print('fin!')

