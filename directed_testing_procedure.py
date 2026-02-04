import pandas as pd
import numpy as np
import torch
import networkx

# import libraries
from multiprocessing import cpu_count
from joblib import Parallel, delayed, parallel_backend
from utils import (parse_args, neg_sampling, seed_everything, 
                   split_dataset, partial_multiplex, scores, sigmoid_predictor,
                   get_node_list, remove_edge_info, split_target_auxiliary_layers,
                   add_edge_info, get_reciprocals, sample_not_reciprocals, find_reciprocals)
import json
import os
from time import time
from copy import deepcopy
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn
import torch
import random
import importlib.util
import sys
import argparse
import pickle
import joblib
#import tensorflow


# import model classes
from models.mell import mell
from models.liamne import liamne
from models.rmne import rmne

EMB_SIZE = 16

def undirected_test_procedure(model_dict, edgetuple_list: list, target_layer: int, sampling_method: str = "uniform", seed = 1234):
    seed_everything(seed)
    node_list = get_node_list(edgetuple_list)
    target_edgetuple_list, aux_edgetuple_list = split_target_auxiliary_layers(edgetuple_list, target_layer)
    train_edge_list, val_edge_list, test_edge_list = split_dataset(remove_edge_info(target_edgetuple_list), split=[.8, .0, .2])
    train_network_edgelist = partial_multiplex(train_edge_list, aux_edgetuple_list, target_layer)
    neg_samples = neg_sampling(train_edge_list + val_edge_list + test_edge_list, node_list, len(test_edge_list), sampling_method=sampling_method)
    test_samples = add_edge_info(test_edge_list, target_layer, 1) + add_edge_info(neg_samples, target_layer, 0)
    random.shuffle(test_samples)
    true_values = [edge[3] for edge in test_samples]

    print(f"Number of negative samples: {len(neg_samples)}, number of reciprocals: {find_reciprocals(remove_edge_info(target_edgetuple_list), neg_samples)}, layer density: {len(target_edgetuple_list)/(len(node_list) * (len(node_list) - 1))}")

    for model_name in model_dict:
        emb_object = model_dict[model_name]({"weighted": False, "directed": True, "emb_size": EMB_SIZE, "node_num": len(node_list), 
                                             "layer_num": 3, "dataset": "Name", "run": 1, "target_layer":1})
        
        emb_object.fit(train_network_edgelist)

        source_emb, target_emb = emb_object.model_return()

        predictions = sigmoid_predictor(source_emb.to_numpy(), target_emb.to_numpy(), test_samples)
        results = scores(true_values, predictions)

        print(results)

def directed_test_procedure(model_dict, edgetuple_list: list, target_layer: int, seed = 1234):
    seed_everything(seed)

    node_list = get_node_list(edgetuple_list)
    target_edgetuple_list, aux_edgetuple_list = split_target_auxiliary_layers(edgetuple_list, target_layer)
    train_edge_list, val_edge_list, test_edge_list = split_dataset(remove_edge_info(target_edgetuple_list), split=[.8, .0, .2])
    train_network_edgelist = partial_multiplex(train_edge_list, aux_edgetuple_list, target_layer)
    reciprocal_samples = get_reciprocals(edgetuple_list)
    not_reciprocal_samples = sample_not_reciprocals(edgetuple_list, node_list, len(test_edge_list))
    test_recip_samples = add_edge_info(test_edge_list, target_layer, 1) + add_edge_info(reciprocal_samples, target_layer, 0)
    test_not_recip_samples = add_edge_info(test_edge_list, target_layer, 1) + add_edge_info(not_reciprocal_samples, target_layer, 0)
    random.shuffle(test_recip_samples)
    true_recip_values = [edge[3] for edge in test_recip_samples]

    print(f"Number of reciprocal samples: {len(reciprocal_samples)}, number of reciprocals: {find_reciprocals(remove_edge_info(target_edgetuple_list), remove_edge_info(reciprocal_samples))}, layer density: {len(target_edgetuple_list)/(len(node_list) * (len(node_list) - 1))}")

    for model_name in model_dict:
        emb_object = model_dict[model_name]({"weighted": False, "directed": True, "emb_size": EMB_SIZE, "node_num": len(node_list), 
                                             "layer_num": 3, "dataset": "Name", "run": 1, "target_layer":1})
        
        emb_object.fit(train_network_edgelist)

        source_emb, target_emb = emb_object.model_return()

        predictions = sigmoid_predictor(source_emb.to_numpy(), target_emb.to_numpy(), test_recip_samples)
        results = scores(true_recip_values, predictions)

        print(results)

    random.shuffle(test_not_recip_samples)
    true_not_recip_values = [edge[3] for edge in test_not_recip_samples]

    print(f"Number of non reciprocal samples: {len(not_reciprocal_samples)}, number of reciprocals: {find_reciprocals(remove_edge_info(target_edgetuple_list), remove_edge_info(not_reciprocal_samples))}, layer density: {len(target_edgetuple_list)/(len(node_list) * (len(node_list) - 1))}")

    for model_name in model_dict:
        emb_object = model_dict[model_name]({"weighted": False, "directed": True, "emb_size": EMB_SIZE, "node_num": len(node_list), 
                                             "layer_num": 3, "dataset": "Name", "run": 1, "target_layer":1})

        predictions = sigmoid_predictor(source_emb.to_numpy(), target_emb.to_numpy(), test_not_recip_samples)
        results = scores(true_not_recip_values, predictions)

        print(results)







if __name__ == "__main__":
    model_dict = {"liamne": liamne, "rmne": rmne, "mell": mell}

    dataframe = pd.read_csv("Vickers-Chan-7thGraders_multiplex.txt", sep=" ", header=None)

    data = list(dataframe.itertuples(index=False, name=None))

    #undirected_test_procedure(model_dict, data, target_layer=1, sampling_method="uniform")

    directed_test_procedure(model_dict, data, target_layer=1)

    print("fin")