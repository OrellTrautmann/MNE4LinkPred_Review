import numpy as np
import random
import argparse
import os
import torch
import pandas as pd
from sklearn.metrics import precision_score, roc_auc_score
from sampling_prob_simulation import uniform_neg_sampling, degree_neg_sampling
import networkx as nx
   
def neg_sampling(edgetuple_list: list, node_list: list, sample_size: int, seed: int = 0, sampling_method: str = 'uniform'):
    if sampling_method == 'uniform':
        neg_samples = uniform_neg_sampling(edgetuple_list, node_list, sample_size)
    elif sampling_method == 'degree':
        neg_samples = degree_neg_sampling(edgetuple_list, node_list, sample_size)
    else:
        return ValueError('Can only implement \'uniform\' or \'degree\'')
    return edgetuple_list + neg_samples, ([1] * len(edgetuple_list)) + ([0] * len(neg_samples))

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


def partial_multiplex(train_edge_list: list, multiplex_aux_edge_list: list, target_layer: int):
    train_edge_list = [(target_layer,) + edge + (1,) for edge in train_edge_list]
    return multiplex_aux_edge_list + train_edge_list

def scores(y_true: list, y_pred: list): 
    score_dict = dict()
    #precision
    pr_scr = precision_score(y_true, y_pred)
    #area under the reciever-operator-curve
    auroc = roc_auc_score(y_true, y_pred)
        
    score_dict.update({'precision' : pr_scr,  'AUROC' : auroc})
    return(score_dict)


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
        print(source_emb[:,0].shape)
        return [np.round(1 / (1 + np.exp(-np.dot(source_emb[:,edge[0]-1], target_emb[:,edge[1]-1])))) for edge in edgetuple_list]