import pandas as pd
from utils import (parse_args, 
                   neg_sampling, 
                   seed_everything, 
                   split_dataset, 
                   construct_training_multiplex, 
                   scores, 
                   sigmoid_predictor,
                   get_node_list,  
                   split_target_auxiliary_layers,
                   get_reciprocals_sample,
                   sample_not_reciprocals, 
                   get_num_layers)

import random
import numpy as np
from tqdm import tqdm
import json
from joblib import Parallel, delayed, cpu_count
import os

def undirected_test_procedure_targetlayer(model_dict: dict, 
                              edgetuple_list: list, 
                              target_layer: int = 1, 
                              emb_size: int = 16,  
                              run: int = 1,
                              network_name = "Vickers",
                              sampling_method: str = "uniform", 
                              test_size = .2,
                              folder: str = '',
                              seed = 1234):
    
    # generate reproducible random seeds 
    seed_everything(seed)
    seeds = np.random.randint(0, 10**6, 4)

    # construct training network
    target_edgetuple_list, aux_edgetuple_list = split_target_auxiliary_layers(edgetuple_list, 
                                                                              target_layer)
    train_edge_list, test_edge_list = split_dataset(target_edgetuple_list, 
                                                    test_size=test_size, 
                                                    seed = seeds[0])
    train_network_edgelist = construct_training_multiplex(train_edge_list, 
                                                        aux_edgetuple_list)

    node_list = get_node_list(edgetuple_list)
    neg_samples = neg_sampling(target_edgetuple_list, 
                               node_list, 
                               len(test_edge_list), 
                               sampling_method=sampling_method,
                               seed = seeds[1])

    test_samples = test_edge_list + neg_samples
    seed_everything(seeds[2])
    random.shuffle(test_samples)
    true_values = [edge[3] for edge in test_samples]


    for model_name in tqdm(model_dict):
        for_json_file = dict(layer = target_layer,
                            run = run,
                            network_name = network_name,
                            model = model_name)

        emb_object = model_dict[model_name]({"emb_size": emb_size, 
                                             "node_num": len(node_list), 
                                             "layer_num": get_num_layers(edgetuple_list),  
                                             "target_layer": target_layer,
                                             "dataset": network_name, 
                                             "run": run,
                                             "seed": seeds[3]})
        
        emb_object.fit(train_network_edgelist)
        source_emb, target_emb = emb_object.model_return()

        predictions = list(map(int, sigmoid_predictor(source_emb.to_numpy(), 
                                                    target_emb.to_numpy(), 
                                                    test_samples)))
        
        results = scores(true_values, predictions)

        for_json_file.update(results)

        # write json file with results
        with open(folder + f"/undirected_test_procedure_results_model_{model_name}_layer{target_layer}_run{run}_seed{seed}.json", "w", encoding="utf-8") as json_file:
            json.dump(for_json_file, json_file, ensure_ascii=False, indent=4)


def directed_test_procedure_targetlayer(model_dict: dict, 
                            edgetuple_list: list, 
                            target_layer: int = 1, 
                            emb_size: int = 16,  
                            run: int = 1,
                            network_name = "Vickers",
                            sampling_method: str = "uniform", 
                            test_size: float = .2,
                            folder: str = '',
                            seed = 1234):
    
    # generate reproducible random seeds 
    seed_everything(seed)
    seeds = np.random.randint(0, 10**6, 6)

    # construct training network
    target_edgetuple_list, aux_edgetuple_list = split_target_auxiliary_layers(edgetuple_list, 
                                                                              target_layer)
    train_edge_list, test_edge_list = split_dataset(target_edgetuple_list, 
                                                    test_size=test_size,
                                                    seed = seeds[0])
    train_network_edgelist = construct_training_multiplex(train_edge_list, 
                                                        aux_edgetuple_list)

    node_list = get_node_list(edgetuple_list)

    reciprocal_samples, true_samples = get_reciprocals_sample(target_edgetuple_list, len(test_edge_list), seed = seeds[1])
    test_recip_samples = true_samples + reciprocal_samples
    seed_everything(seeds[2])
    random.shuffle(test_recip_samples)
    true_recip_values = [edge[3] for edge in test_recip_samples]

    not_reciprocal_samples = sample_not_reciprocals(edgetuple_list, node_list, len(test_edge_list), seed = seeds[4])
    test_not_recip_samples = test_edge_list + not_reciprocal_samples
    seed_everything(seeds[2])
    random.shuffle(test_not_recip_samples)
    true_not_recip_values = [edge[3] for edge in test_not_recip_samples]


    # iterate test over the models
    for model_name in tqdm(model_dict):
        for_recip_json_file = dict(layer = target_layer,
                                    run = run,
                                    network_name = network_name,
                                    model=model_name)
    
        for_not_recip_json_file = dict(layer = target_layer,
                                        run = run,
                                        network_name = network_name,
                                        model=model_name)

        emb_object = model_dict[model_name]({"emb_size": emb_size, 
                                             "node_num": len(node_list), 
                                             "layer_num": get_num_layers(edgetuple_list),  
                                             "target_layer": target_layer,
                                             "dataset": network_name, 
                                             "run": run,
                                             "seed": seeds[3]})
        
        emb_object.fit(train_network_edgelist)
        source_emb, target_emb = emb_object.model_return()

        recip_predictions = sigmoid_predictor(source_emb.to_numpy(), 
                                        target_emb.to_numpy(), 
                                        test_recip_samples)
        recip_results = scores(true_recip_values, recip_predictions)

        for_recip_json_file.update(recip_results)

        not_recip_predictions = sigmoid_predictor(source_emb.to_numpy(), 
                                        target_emb.to_numpy(), 
                                        test_not_recip_samples)
        not_recip_results = scores(true_not_recip_values, not_recip_predictions)

        for_not_recip_json_file.update(not_recip_results)

        # write json file with results
        with open(folder + f"/Reciprocals/reciprocal_directed_test_procedure_results_model_{model_name}_layer{target_layer}_run{run}_seed{seed}.json", "w", encoding="utf-8") as json_file:
            json.dump(for_recip_json_file, json_file, ensure_ascii=False, indent=4)

        with open(folder + f"/NotReciprocals/not_reciprocal_directed_test_procedure_results_model_{model_name}_layer{target_layer}_run{run}_seed{seed}.json", "w", encoding="utf-8") as json_file:
            json.dump(for_not_recip_json_file, json_file, ensure_ascii=False, indent=4)


def parallel_work(model_dict, 
                            edgetuple_list, 
                            emb_size,
                            run,
                            run_seed,
                            network_name,
                            sampling_method, 
                            test_size,
                            path_undirected,
                            path_directed,
                            num_layers):
            
            seed_everything(run_seed)
            new_seeds = np.random.randint(0, 10**6, num_layers)

            for target_layer in range(1, num_layers + 1):
                undirected_test_procedure_targetlayer(model_dict = model_dict, 
                                                    edgetuple_list = edgetuple_list, 
                                                    emb_size = emb_size,  
                                                    run = run,
                                                    target_layer = target_layer,
                                                    network_name = network_name,
                                                    sampling_method = sampling_method, 
                                                    test_size=test_size,
                                                    folder=path_undirected,
                                                    seed = new_seeds[target_layer-1])
                
                directed_test_procedure_targetlayer(model_dict = model_dict, 
                                                    edgetuple_list = edgetuple_list, 
                                                    emb_size = emb_size,  
                                                    run = run,
                                                    target_layer = target_layer,
                                                    network_name = network_name,
                                                    sampling_method = sampling_method, 
                                                    test_size=test_size,
                                                    folder=path_directed,
                                                    seed = new_seeds[target_layer-1])

def test_procedure(model_dict: dict, 
                   edgetuple_list: list, 
                   emb_size: int = 16,  
                   runs: int = 1,
                   network_name = "Missing_name",
                   sampling_method: str = "uniform", 
                   test_size:float = .2,
                   outdir: str = "Results",
                   njobs: int = 1,
                   seed = 1234):
    
    seed_everything(seed)
    seeds = np.random.randint(0, 10**6, runs)
    
    num_layers = get_num_layers(edgetuple_list)

    path_undirected = outdir + '/' + network_name + '/Undirected'
    path_directed = outdir + '/' + network_name + '/Directed'

    Parallel(n_jobs=njobs, backend="loky")(delayed(parallel_work)(model_dict, 
                                                        edgetuple_list, 
                                                        emb_size,
                                                        run,
                                                        run_seed,
                                                        network_name,
                                                        sampling_method, 
                                                        test_size,
                                                        path_undirected,
                                                        path_directed,
                                                        num_layers) for run, run_seed in enumerate(seeds))

def evaluation(outdir: str, network_name: str, metrics: list):
    subdir_list = ['/Undirected', '/Directed/NotReciprocals', '/Directed/Reciprocals']
    df_list = []
    for subdir in subdir_list:
        final_stat_dict_list = list()
        path = outdir + '/' + network_name + subdir
        results = list()
        for file in os.listdir(path):
            with open(path + '/' + file, 'r', encoding="utf-8") as json_file:
                file_results = json.load(json_file)
                results.append(file_results)
        
        results_df = pd.DataFrame.from_records(results)
        stat_df = results_df.groupby(["layer", "model"]).describe().round(3)

        for index in stat_df.index:
            temp_dict = {'index': index}
            for metric in metrics:
                mean_std_df = stat_df.loc[index,metric][["mean", "std"]].astype(str)
                mean_std_dict = mean_std_df.to_dict()
                temp_dict.update({metric: mean_std_dict["mean"] + " (" + mean_std_dict["std"] + ")"})

            final_stat_dict_list.append(temp_dict)

        print(subdir)
        print(pd.DataFrame(final_stat_dict_list).to_latex(index=False))

def edgetuplelist_from_csvfile(file_path: str):
    dataframe = pd.read_csv(file_path, sep=" ", header=None)
    return list(dataframe.itertuples(index=False, name=None))


if __name__ == "__main__":
    # import model classes
    from models.mell import mell
    from models.liamne import liamne
    from models.rmne import rmne

    if cpu_count() > 2:
        njobs = cpu_count() - 2
    else:
        njobs = 1

    print("njobs",njobs)

    model_dict = {"mell": mell, "liamne": liamne, "rmne": rmne}
    metric_list = ["AUROC",
                   "accuracy",
                   "avg_prec"]

    args = parse_args()

    file_path = args.inpath
    data = edgetuplelist_from_csvfile(file_path)
    network_name = file_path.split('/')[1]

    outdir = args.outdir
    if not os.path.exists(outdir + '/' + network_name + '/Undirected'):
        os.makedirs(outdir + '/' + network_name + '/Undirected')
    if not os.path.exists(outdir + '/' + network_name + '/Directed/NotReciprocals'):
        os.makedirs(outdir + '/' + network_name + '/Directed/NotReciprocals')
    if not os.path.exists(outdir + '/' + network_name + '/Directed/Reciprocals'):
        os.makedirs(outdir + '/' + network_name + '/Directed/Reciprocals')

    test_procedure(model_dict, data, emb_size=args.dim, runs = args.runs, network_name = network_name, sampling_method="uniform", test_size=args.testsize, outdir=outdir, njobs = njobs, seed = args.seed)

    evaluation(outdir=outdir, network_name=network_name, metrics=metric_list)

    print("fin")
