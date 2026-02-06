import numpy as np
import random
import argparse
import os
import torch
from sklearn.metrics import (precision_score, 
                             roc_auc_score, 
                             accuracy_score, 
                             average_precision_score)

from sampling_prob_simulation import (uniform_neg_sampling, 
                                      degree_neg_sampling)

import networkx as nx
import itertools
import json
   
def neg_sampling(edgetuple_list: list, node_list: list, sample_size: int, seed: int = 0, sampling_method: str = 'uniform'):
    if sampling_method == 'uniform':
        neg_samples = uniform_neg_sampling(remove_edge_info(edgetuple_list), node_list, sample_size)
    elif sampling_method == 'degree':
        neg_samples = degree_neg_sampling(remove_edge_info(edgetuple_list), node_list, sample_size)
    else:
        return ValueError('Can only implement \'uniform\' or \'degree\'')
    return add_edge_info(neg_samples, edgetuple_list[0][0], weight=0)

def split_dataset(edgetuple_list: list, split=[.7, .1, .2]):
    graph = nx.Graph(edgetuple_list) # is undirected
    minimal_tree = nx.minimum_spanning_tree(graph) # insure graph is still connected
    train_edge_list = list(set(edgetuple_list).intersection(set(minimal_tree.edges)))
    remain_edge_list = list(set(edgetuple_list).difference(set(minimal_tree.edges)))
    if len(train_edge_list) / len(edgetuple_list) < split[0]:
        train_edge_list.extend(random.sample(remain_edge_list, round(split[0] * len(edgetuple_list) - len(train_edge_list))))
    elif len(train_edge_list) / len(edgetuple_list) > split[0]:
        return ValueError('The graph does not contain enough connected edges for split')
    
    val_edge_list = random.sample(remain_edge_list, round(split[1] / (split[1] + split[2]) * len(remain_edge_list)))
    test_edge_list = list(set(remain_edge_list).difference(set(val_edge_list)))

    return train_edge_list, val_edge_list, test_edge_list

def split_target_auxiliary_layers(edgetuple_list: list, target_layer: int):
    target_layer_edge_list = list()
    aux_layers_edge_list = list()
    for edge in edgetuple_list:
        if edge[0] == target_layer:
            target_layer_edge_list.append(edge)
        else:
            aux_layers_edge_list.append(edge)
    return target_layer_edge_list, aux_layers_edge_list

def remove_edge_info(edgetuple_list: list):
    return [(edge[1], edge[2]) for edge in edgetuple_list]

def add_edge_info(edgetuple_list: list, target_layer: int, weight: int = 1):
    return [(target_layer, edge[0], edge[1], weight) for edge in edgetuple_list]

def get_node_list(edgetuple_list: list):
    reduced_edgetuples = [(edge[1], edge[2]) for edge in edgetuple_list]
    return list(set(itertools.chain.from_iterable(reduced_edgetuples)))

def partial_multiplex(train_edge_list: list, multiplex_aux_edge_list: list, target_layer: int):
    train_edge_list = [(target_layer,) + edge + (1,) for edge in train_edge_list]
    return multiplex_aux_edge_list + train_edge_list

def get_reciprocals(edgetuple_list: list):
    reciprocals = list()
    for edge in edgetuple_list:
        if (edge[0], edge[2], edge[1], edge[3]) in edgetuple_list:
            continue
        else:
            reciprocals.append((edge[0], edge[2], edge[1], 0))
    return reciprocals

def sample_not_reciprocals(edgetuple_list: list, node_list: list, sample_size: int, seed: int = 0):
    neg_samples = list()
    first_edge = edgetuple_list[0]

    for _ in range(sample_size):
        edge = (first_edge[0], np.random.randint(1,max(node_list)+1), np.random.randint(1,max(node_list)+1), 1)
        reciprocal_edge = (edge[0], edge[2], edge[1], edge[3])
        while edge in edgetuple_list or reciprocal_edge in edgetuple_list:
            edge = (first_edge[0], np.random.randint(1,max(node_list)+1), np.random.randint(1,max(node_list)+1), 1)
            reciprocal_edge = (edge[0], edge[2], edge[1], edge[3])
        
        neg_samples.append(edge)
    return neg_samples

def find_reciprocals(edgetuple_list: list, neg_edgetuple_list: list):
    count = 0
    for (source, target) in neg_edgetuple_list:
        if (target, source) in edgetuple_list:
            count += 1
    return count

def retrieve_true_edge(neg_samples):
    return [(edge[2], edge[1]) for edge in neg_samples]

def write_2_json(file_path_name: str, new_data: dict):
    with open(file_path_name, 'r+') as file:
        # Load existing data into a dictionary
        file_data = json.load(file)
        
        # Append new data to the 'emp_details' list
        file_data.append(new_data)
        
        # Move the cursor to the beginning of the file
        file.seek(0)
        
        # Write the updated data back to the file
        json.dump(file_data, file)

def scores(y_true: list, y_pred: list): 
    #precision
    pr_scr = precision_score(y_true, y_pred)
    #area under the reciever-operator-curve
    auroc = roc_auc_score(y_true, y_pred)
    # accuracy
    acc_scr = accuracy_score(y_true, y_pred)
    # average precision score
    ap_scr = average_precision_score(y_true, y_pred, average='macro')
        
    score_dict = dict(precision = pr_scr,
                      AUROC = auroc,
                      accuracy = acc_scr,
                      avg_prec = ap_scr)
    return(score_dict)

def get_num_layers(edgetuple_list: list):
    return max(list(itertools.chain.from_iterable(edgetuple_list))[::4])


def parse_args():
    parser = argparse.ArgumentParser(description='Multiplex network embedding tester pipeline')
    parser.add_argument('-m', '--model', type=str, default='liamne',
    					help='Model class written in python')
    parser.add_argument('-p', '--modelpath', type=str, default='liamne.py',
    					help='path to file with model class written in python')
    parser.add_argument('-d', '--dim', type=int, default=16,
    					help='size of node embeddings')
    parser.add_argument('-l', '--layer', type=int, default=1,
    					help='layer to test on')
    parser.add_argument('-w', '--weighted', type=int, default=0,
    					help='is the graph weighted or not')
    parser.add_argument('-g', '--directed', type=int, default=0,
    					help='is the graph directed or not')
    parser.add_argument('-r', '--runs', type=int, default=1,
    					help='number of runs')
    parser.add_argument('-i', '--inpath', type=str, default='Datasets/Vickers/Vickers-Chan-7thGraders_multiplex.txt',
    					help='path to the dataset')
    parser.add_argument('-o', '--outdir', type=str, default='RESULTS',
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
        print(source_emb[:,0])
        return [np.round(1 / (1 + np.exp(-np.dot(source_emb[:,edge[0]-1], target_emb[:,edge[1]-1])))) for edge in edgetuple_list]