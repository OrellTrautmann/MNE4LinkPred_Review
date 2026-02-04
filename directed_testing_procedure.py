import pandas as pd
import numpy as np
import torch
import networkx

# import libraries
from multiprocessing import cpu_count
from joblib import Parallel, delayed, parallel_backend
from utils import parse_args, neg_sampling, seed_everything, split_dataset, partial_multiplex, scores, sigmoid_predictor
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

model_dict = {"liamne": liamne, "rmne": rmne, "mell": mell}

"""
def model_pipeline(data_path, train, val, test, train_network_path, layers: list, target_layer: int, dim: int, weighted: bool, directed: bool, data_name: str, model_name: str, nodes: list, run: int, seed=0):
    #### assertions ########
    assert (isinstance(train, list))
    assert (isinstance(test, list))
    assert (isinstance(val, list))
    # assert(model_name in model_dict.keys())
    assert (dim > 0)
    assert (len(nodes) > 0)
    assert (seed >= 0)
    ########################
    # load datas
    data = pd.read_csv(data_path, sep=" ", header=None)
    train_network = pd.read_csv(train_network_path, sep=" ", header=None)
    print(train_network)
    # generate seeds
    seed_everything(seed)
    seeds = np.random.choice(int(1e5), size=2)

    ###### Embedding #######
    emb_object = model_dict[model_name]({"node_num": len(nodes), "layer_num": len(
        layers), "emb_size": dim, "target_layer": target_layer, "weighted": weighted, "directed": directed, "dataset": data_name, "run": run, "seed": seeds[0]})
    start_time = time()
    emb_object.fit(train_network)
    run_time = time() - start_time
    embeddings = emb_object.model_return()
    ########################

    ###### Prediction ######
    X_input = reg_clf_input(data[[1, 2]], embeddings).sort_index()

    y_input = data[[3]].sort_index()

    
    ########################

    return {"results": scores(np.array(y_input.loc[test]), np.array(y_pred), weighted=weighted), "run time": run_time}


def aux_pipeline(data_path: str, layers: list, layer: int, dim: int, weighted: bool, directed: bool, dataset_name: str, nodes: list, run: int, model_name: str, seed: int):
    
    ###### assertions #####
    assert (dim > 0)
    assert (run >= 0)
    assert (layer >= 1)
    assert (seed >= 0)
    #######################

    # generate seeds
    seed_everything(seed)
    seeds = list(np.random.choice(int(1e5), 2))

    ##### Dataset split ####
    train, val, test = split_dataset_indices(data_path, layer, seeds[0])
    data = pd.read_csv(data_path, sep=" ", header=None)
    
    train_network_path = partial_multiplex(train, layer, data_path, model_name, layer, run, dataset_name)
    ########################

    results = {f"{model_name}_{layer}_{run}": model_pipeline(data_path, train, val, test, train_network_path, layers, layer, dim, weighted, directed, dataset_name, model_name, nodes, run, seed=seeds[1])}

    return results
    

def run_pipeline(model_name: str, file_path: str, dim: int, weighted: bool, directed: bool, runs: int, layer: int, outdir: str, njobs: int):
    ##### assertions ########
    assert (dim > 0)
    assert (runs > 0)
    assert (njobs >= 1)
    #########################

    file_name = file_path.split('/')[-1]
    writepath = outdir + f"/{model_name}_{layer}_{file_name.split('.')[0]}"

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data = pd.read_csv(file_path, sep=" ", header=None)
    layers = list(data[0].unique())
    nodes = list(set(data[1]).union(set(data[2])))

    # generate random seeds
    seed_everything(1234)
    seeds = list(np.random.choice(int(1e5), size=runs+len(layers)))

    # use negative sampling
    layer_sample_lst = [neg_sampling(data[data[0] == target_layer], target_layer, seeds[target_layer-1], directed=directed) for target_layer in layers]
    data = pd.concat(layer_sample_lst, axis=0, ignore_index=True)
    print("negsampl")
    if not os.path.exists("NegSampledDataSet/"):
        os.makedirs("NegSampledDataSet/")

    data_path = "NegSampledDataSet/" + \
        f"{file_name}_{model_name}_{layer}_{file_name.split('.')[0]}.csv"
    
    data.to_csv(data_path, sep=" ", header=False, index=False)
    
    run_seed_list = enumerate(seeds[len(layers):])
    assert (len(seeds[len(layers):]) == runs)

    if (njobs == 1) or (runs == 1):

        for run, seed in run_seed_list:
            # run through each layer as a target layer and take the average performance
            results = aux_pipeline(data_path, layers, layer, dim, weighted,
                                   directed, file_name, nodes, run+1, model_name, seed)

            with open(writepath + f"_seed={seed}_run={run}.txt", 'w') as f:
                f.write(json.dumps(results))

    else:
        def parallel_work(data_path, layers, layer, dim, weighted, directed, file_name, nodes, run, model_name, seed, writepath):
            print("work")
            results = aux_pipeline(data_path, layers, layer, dim, weighted,directed, file_name, nodes, run, model_name, seed)
            
            with open(writepath  + f"_seed={seed}_run={run}.txt", 'w') as f:
                f.write(json.dumps(results))

        with parallel_backend('threading', n_jobs=njobs):
            print("para")
            Parallel()(delayed(parallel_work)(data_path, layers, layer, dim, weighted, directed, file_name, nodes, run+1, model_name, seed, writepath) for run, seed in run_seed_list)

    print("x")


if __name__ == "__main__":


    # Datafiles for form: (file path, embedding dimenstion, is weighted, is directed)
    real_world_files = [
        ("Datasets/Vickers/Vickers-Chan-7thGraders_multiplex.txt", 16, False, True), #check
        #("Datasets/Twitter/Twitter.csv", 128, True, True),
        #("Datasets/SacchCere/sacchcere_genetic_multiplex.edges.txt", 64, False, True), 
        #("Datasets/Leukemia/Leukemia MP 1.csv", 128, False, False), 
        #("Datasets/Drug/DrugMultiplex.csv", 64, True, False), 
        ("Datasets/CKM/CKM-Physicians-Innovation_multiplex.txt", 16, False, True) #check
        ]

    if cpu_count() > 2:
        njobs = cpu_count() - 2
    else:
        njobs = 1

    print("njobs: %d" % njobs)

    args = parse_args()
    print("args")
    if not args.model in model_dict:

        spec = importlib.util.spec_from_file_location(
            args.model, args.modelpath)
        foo = importlib.util.module_from_spec(spec)
        sys.modules[args.model] = foo
        spec.loader.exec_module(foo)

        model_dict.update({args.model: foo.MyModelClass})

        run_pipeline(args.model, args.inpath, args.dim, bool(args.weighted), bool(
            args.directed), args.runs, args.layer, args.outdir, njobs)

    run_pipeline(args.model, args.inpath, args.dim, bool(args.weighted), bool(
        args.directed), args.runs, args.layer, args.outdir, njobs)
"""

if __name__ == "__main__":

    dataframe = pd.read_csv("Vickers-Chan-7thGraders_multiplex.txt", sep=" ", header=None)

    data = list(dataframe.itertuples(index=False, name=None))

    model = mell({"weighted": False, "directed": True, "emb_size": 16, "node_num": 29, "layer_num": 3, "dataset": "Vickers", "run": 1, "target_layer":1})

    model.fit(data)

    S, T = model.model_return()

    print(sigmoid_predictor(S.to_numpy(), T.to_numpy(), data))

    print("fin")