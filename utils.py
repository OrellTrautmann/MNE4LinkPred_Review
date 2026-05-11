import numpy as np
import random
import argparse
import os
import torch
from sklearn.metrics import (precision_score, 
                             roc_auc_score, 
                             accuracy_score, 
                             average_precision_score)

import networkx as nx
import itertools
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_node_list(edgetuple_list: list):
    reduced_edgetuples = [(edge[1], edge[2]) for edge in edgetuple_list]
    return list(set(itertools.chain.from_iterable(reduced_edgetuples)))

def remove_edge_info(edgetuple_list: list):
    return [(edge[1], edge[2]) for edge in edgetuple_list]

def neg_sampling(edgetuple_list: list, node_list: list, sample_size: int, seed: int = 1234, sampling_method: str = 'uniform'):
    if sampling_method == 'uniform':
        neg_samples = uniform_neg_sampling(edgetuple_list, node_list, sample_size, seed = seed)
    elif sampling_method == 'degree':
        neg_samples = degree_neg_sampling(edgetuple_list, node_list, sample_size, seed = seed)
    else:
        return ValueError('Can only implement \'uniform\' or \'degree\'')
    return neg_samples

def split_dataset(edgetuple_list: list, test_size = .2, seed = 1234):
    seed_everything(seed)
    graph = nx.Graph(remove_edge_info(edgetuple_list)) # is undirected
    minimal_tree = nx.minimum_spanning_tree(graph) # insure graph is still connected
    train_edge_list = list(set(edgetuple_list).intersection(set(minimal_tree.edges)))
    remain_edge_list = list(set(edgetuple_list).difference(set(minimal_tree.edges)))
    if len(train_edge_list) / len(edgetuple_list) < 1 - test_size:
        additional_edge_list = random.sample(remain_edge_list, round((1 - test_size) * len(edgetuple_list) - len(train_edge_list)))
        train_edge_list.extend(additional_edge_list)
        remain_edge_list = list(set(remain_edge_list).difference(set(additional_edge_list)))
    elif len(train_edge_list) / len(edgetuple_list) > 1 - test_size:
        return ValueError('The graph does not contain enough connected edges for split')

    return train_edge_list, remain_edge_list

def edgelist_2_edgedict(edgetuple_list):
    num_layers = get_num_layers(edgetuple_list)
    layer_dict_edge_list = dict()

    for layer in range(1, num_layers + 1):
        layer_dict_edge_list.update({layer : []})

    for edge in edgetuple_list:
        layer_dict_edge_list[edge[0]].append((edge[1], edge[2]))
    
    return layer_dict_edge_list

def edgelist_from_edgedict(edgetuple_dict: dict, weight: int):
    edgetuple_list = list()
    for layer in edgetuple_dict.keys():
        edgetuple_list.extend([(layer, edge[0], edge[1], weight) for edge in edgetuple_dict])
    return edgetuple_list

def split_target_auxiliary_layers(edgetuple_list: list, target_layer: int):
    target_layer_edge_list = list()
    aux_layers_edge_list = list()
    for edge in edgetuple_list:
        if edge[0] == target_layer:
            target_layer_edge_list.append(edge)
        else:
            aux_layers_edge_list.append(edge)
    return target_layer_edge_list, aux_layers_edge_list

def construct_training_multiplex(train_edge_list: list, multiplex_aux_edge_list: list):
    return multiplex_aux_edge_list + train_edge_list

def get_reciprocals(edgetuple_list: list):
    reciprocals = list()
    true_edges = list()
    for edge in edgetuple_list:
        if (edge[0], edge[2], edge[1], edge[3]) in edgetuple_list:
            continue
        else:
            reciprocals.append((edge[0], edge[2], edge[1], 0))
            true_edges.append(edge)
    return reciprocals, true_edges

def get_reciprocals_sample(edgetuple_list: list, sample_size: int, seed: int = 1234):
    seed_everything(seed)
    reciprocals, true_edges = get_reciprocals(edgetuple_list)
    if len(reciprocals) > sample_size:
        indices = np.random.choice(len(reciprocals), size= len(reciprocals) - sample_size, replace=False)
        print(indices)
        return [reciprocals[index] for index in indices], [true_edges[index] for index in indices]
    else:
        return reciprocals, true_edges
                         
def sample_not_reciprocals(edgetuple_list: list, node_list: list, sample_size: int, seed: int = 1234):
    seed_everything(seed)
    neg_samples = list()
    first_edge = edgetuple_list[0]

    for _ in range(sample_size):
        edge = (first_edge[0], np.random.randint(1,max(node_list)+1), np.random.randint(1,max(node_list)+1), 1)
        reciprocal_edge = (edge[0], edge[2], edge[1], edge[3])
        while edge in edgetuple_list or reciprocal_edge in edgetuple_list:
            edge = (first_edge[0], np.random.randint(1,max(node_list)+1), np.random.randint(1,max(node_list)+1), 1)
            reciprocal_edge = (edge[0], edge[2], edge[1], edge[3])
        
        neg_samples.append((edge[0], edge[1], edge[2], 0))
    return neg_samples

def find_reciprocals(edgetuple_list: list, neg_edgetuple_list: list):
    count = 0
    for edge in neg_edgetuple_list:
        if (edge[0], edge[2], edge[1], 1) in edgetuple_list:
            count += 1
    return count

def retrieve_true_edge(neg_samples):
    return [(edge[0], edge[2], edge[1], 1) for edge in neg_samples]

def read_from_json(file_path_name: str):
    try:
        result_dict = json.loads(file_path_name)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
    return result_dict

def scores(y_true: list, y_pred: list): 
    
    auroc = roc_auc_score(y_true, y_pred)
    
    acc_scr = accuracy_score(y_true, y_pred)
    
    ap_scr = average_precision_score(y_true, y_pred, average='macro')
        
    score_dict = dict(AUROC = auroc,
                      accuracy = acc_scr,
                      avg_prec = ap_scr)
    return(score_dict)

def get_num_layers(edgetuple_list: list):
    return max(list(itertools.chain.from_iterable(edgetuple_list))[::4])

def degree_neg_sampling(edgetuple_list: list, node_list: list, sample_size: int, seed: int = 1234):
    seed_everything(seed)
    outdegree_array = np.zeros((len(node_list),), dtype=np.int32)
    layer = edgetuple_list[0][0]
    for edge in edgetuple_list:
        outdegree_array[edge[1]-1] += 1

    node_probability = outdegree_array / len(edgetuple_list)

    rng = np.random.default_rng()
    node_samples = rng.choice(node_list, size=sample_size, p=node_probability)

    neg_edgetuple_list = list()

    for target in node_samples:
        edge = (layer, np.random.randint(1,max(node_list)+1), target, 1)
        while edge in edgetuple_list or edge in neg_edgetuple_list:
            edge = (layer, np.random.randint(1,max(node_list)+1), target, 1)
        
        neg_edgetuple_list.append((edge[0], edge[1], edge[2], 0))

    return neg_edgetuple_list

def uniform_neg_sampling(edgetuple_list: list, node_list: list, sample_size: int, seed: int = 1234):
    seed_everything(seed)
    layer = edgetuple_list[0][0]
    neg_edgetuple_list = list()
    for _ in range(sample_size):
        edge = (layer, np.random.randint(1,max(node_list)+1), np.random.randint(1,max(node_list)+1), 1)
        while edge in edgetuple_list or edge in neg_edgetuple_list:
            edge = (layer, np.random.randint(1,max(node_list)+1), np.random.randint(1,max(node_list)+1), 1)
        
        neg_edgetuple_list.append((edge[0], edge[1], edge[2], 0))

    return neg_edgetuple_list


def parse_args():
    parser = argparse.ArgumentParser(description='Multiplex network embedding tester pipeline')
    parser.add_argument('-d', '--dim', type=int, default=16,
    					help='size of node embeddings')
    parser.add_argument('-r', '--runs', type=int, default=10,
    					help='number of runs')
    parser.add_argument('-s', '--seed', type=int, default=124,
    					help='random seed value')
    parser.add_argument('-t', '--testsize', type=float, default=.2,
    					help='test set size ratio')
    parser.add_argument('-i', '--inpath', type=str, default='Datasets/Vickers/Vickers-Chan-7thGraders_multiplex.edges.txt',
    					help='path to the dataset')
    parser.add_argument('-o', '--outdir', type=str, default='Results',
    					help='path folder to save the results')
    return parser.parse_args()
    

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)

def sigmoid_predictor(source_emb: np.array, target_emb: np.array, edgetuple_list):
        return [np.round(1 / (1 + np.exp(-np.dot(source_emb[:,edge[1]-1], target_emb[:,edge[2]-1])))) for edge in edgetuple_list]


def compute_statistics(model_layer, dataset_file, data_dict_lst):
    data_df = pd.DataFrame.from_records(data_dict_lst, index="run")
    data_stat_df = data_df.describe().round(3)
    mean_std_df = data_stat_df.loc[["mean", "std", "50%"]].astype(str)
    mean_std_dict = mean_std_df.to_dict()
    new_mean_std_dict = dict()
    for key in mean_std_dict:
        new_mean_std_dict[key] = mean_std_dict[key]["mean"] + " [" + mean_std_dict[key]["50%"] + "]" + " (" + mean_std_dict[key]["std"] + ")"

    final_dict = {"data": dataset_file, "model" : model_layer.split("_")[0], "layer": model_layer.split("_")[-1]}
    final_dict.update(new_mean_std_dict)
    return  final_dict 


def print_performance_table(all_data_dict_lst, outdir, **kwargs):
    df = pd.DataFrame(all_data_dict_lst)
    # Convert DataFrame to LaTeX
    latex_code = df.to_latex(index=False)

    # Print or save the LaTeX code
    print(kwargs)
    print(latex_code)
    
    path = ""
    if kwargs:
        for something in list(kwargs):
            path += f"_{kwargs[something]}"

    # Optionally, save to a .tex file
    with open(f"{outdir}/table{path}.tex", "w") as file:
        file.write(latex_code)
    

def boxplots_per_network_per_score_per_layer(network_lst, score_lst, plotdata_lst, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    data = pd.DataFrame.from_records(plotdata_lst) 
    for network, _, weighted, _, _, _ in network_lst:
        if weighted:
            for score in ["NRMSE", "run time"]:
                sns.boxplot(data=data, x="layer", y=score, hue="model")
                plt.savefig(image_path + f"/{network}_{score}_model_comparison.png")
                plt.close() 
        else:
            for score in score_lst:
                sns.boxplot(data=data, x="layer", y=score, hue="model")
                plt.savefig(image_path + f"/{network}_{score}_model_comparison.png")
                plt.close()                     
    
    return