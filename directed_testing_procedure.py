import pandas as pd
from utils import (parse_args, 
                   neg_sampling, 
                   seed_everything, 
                   split_dataset, 
                   partial_multiplex, 
                   scores, 
                   sigmoid_predictor,
                   get_node_list, 
                   remove_edge_info, 
                   split_target_auxiliary_layers,
                   add_edge_info,  
                   get_reciprocals_sample,
                   sample_not_reciprocals, 
                   find_reciprocals,
                   get_num_layers)

import random
import numpy as np
from tqdm import tqdm
import json
import os

# import model classes
from models.mell import mell
from models.liamne import liamne
from models.rmne import rmne

EMB_SIZE = 16

def undirected_test_procedure_targetlayer(model_dict: dict, 
                              edgetuple_list: list, 
                              target_layer: int = 1, 
                              emb_size: int = 16,  
                              run: int = 1,
                              network_name = "Vickers",
                              sampling_method: str = "uniform", 
                              split = [.8, .0, .2],
                              seed = 1234):
    
    # generate reproducible random seeds 
    seed_everything(seed)
    seeds = np.random.randint(0, 10**6, 4)

    # construct training network
    target_edgetuple_list, aux_edgetuple_list = split_target_auxiliary_layers(edgetuple_list, 
                                                                              target_layer)
    train_edge_list, val_edge_list, test_edge_list = split_dataset(remove_edge_info(target_edgetuple_list), 
                                                                   split=split, 
                                                                   seed = seeds[0])
    train_network_edgelist = partial_multiplex(train_edge_list, 
                                               aux_edgetuple_list, 
                                               target_layer)

    # generate testing set with negative samples
    node_list = get_node_list(edgetuple_list)
    neg_samples = neg_sampling(target_edgetuple_list, 
                               node_list, 
                               len(test_edge_list), 
                               sampling_method=sampling_method,
                               seed = seeds[1])
    test_samples = add_edge_info(test_edge_list, 
                                 target_layer, 
                                 weight = 1) 
    test_samples += neg_samples
    seed_everything(seeds[2])
    random.shuffle(test_samples)
    true_values = [edge[3] for edge in test_samples]

    # initial dict for results to e saved as json file
    for_json_file = dict(number_negative_samples = len(neg_samples), 
                         number_reciprocals = find_reciprocals(remove_edge_info(target_edgetuple_list), 
                                                              remove_edge_info(neg_samples)),
                         layer_density = len(target_edgetuple_list) / (len(node_list) * (len(node_list) - 1)))

    # iterate test over the models
    for model_name in tqdm(model_dict):
        emb_object = model_dict[model_name]({"emb_size": emb_size, 
                                             "node_num": len(node_list), 
                                             "layer_num": get_num_layers(edgetuple_list),  
                                             "target_layer": target_layer,
                                             "dataset": network_name, 
                                             "run": run,
                                             "seed": seeds[3]})
        
        emb_object.fit(train_network_edgelist)
        source_emb, target_emb = emb_object.model_return()

        predictions = sigmoid_predictor(source_emb.to_numpy(), 
                                        target_emb.to_numpy(), 
                                        remove_edge_info(test_samples))
        results = scores(true_values, predictions)

        for_json_file.update({f"{model_name}": results})

    # write json file with results
    with open(f"Results/Undirected/undirected_test_procedure_results_run{run}_seed{seed}.json", "w", encoding="utf-8") as json_file:
        json.dump(for_json_file, json_file, ensure_ascii=False, indent=4)


def directed_test_procedure_targetlayer(model_dict: dict, 
                            edgetuple_list: list, 
                            target_layer: int = 1, 
                            emb_size: int = 16,  
                            run: int = 1,
                            network_name = "Vickers",
                            sampling_method: str = "uniform", 
                            split = [.8, .0, .2],
                            seed = 1234):
    
    # generate reproducible random seeds 
    seed_everything(seed)
    seeds = np.random.randint(0, 10**6, 6)

    # construct training network
    target_edgetuple_list, aux_edgetuple_list = split_target_auxiliary_layers(edgetuple_list, 
                                                                              target_layer)
    train_edge_list, val_edge_list, test_edge_list = split_dataset(remove_edge_info(target_edgetuple_list), 
                                                                   split=split,
                                                                   seed = seeds[0])
    train_network_edgelist = partial_multiplex(train_edge_list, 
                                               aux_edgetuple_list, 
                                               target_layer)

    # test on reciprocals
    node_list = get_node_list(edgetuple_list)

    reciprocal_samples = get_reciprocals_sample(target_edgetuple_list, len(test_edge_list), seed = seeds[1])
    test_recip_samples = add_edge_info(test_edge_list, target_layer, weight = 1) + reciprocal_samples
    seed_everything(seeds[2])
    random.shuffle(test_recip_samples)
    true_recip_values = [edge[3] for edge in test_recip_samples]

    not_reciprocal_samples = sample_not_reciprocals(edgetuple_list, node_list, len(test_edge_list), seed = seeds[4])
    test_not_recip_samples = add_edge_info(test_edge_list, target_layer, weight = 1) + add_edge_info(not_reciprocal_samples, target_layer, weight = 0)
    seed_everything(seeds[5])
    random.shuffle(test_not_recip_samples)
    true_not_recip_values = [edge[3] for edge in test_not_recip_samples]

    # initial dict for results to e saved as json file
    for_recip_json_file = dict(number_negative_samples = len(reciprocal_samples), 
                               number_reciprocals = find_reciprocals(remove_edge_info(target_edgetuple_list), remove_edge_info(reciprocal_samples)),
                               layer_density = len(target_edgetuple_list) / (len(node_list) * (len(node_list) - 1)))
    
    for_not_recip_json_file = dict(number_negative_samples = len(not_reciprocal_samples), 
                                  number_reciprocals = find_reciprocals(remove_edge_info(target_edgetuple_list), remove_edge_info(not_reciprocal_samples)),
                                  layer_density = len(target_edgetuple_list) / (len(node_list) * (len(node_list) - 1)))

    # iterate test over the models
    for model_name in tqdm(model_dict):
        emb_object = model_dict[model_name]({"emb_size": emb_size, 
                                             "node_num": len(node_list), 
                                             "layer_num": get_num_layers(edgetuple_list),  
                                             "target_layer": target_layer,
                                             "dataset": network_name, 
                                             "run": run,
                                             "seed": seeds[3]})
        
        emb_object.fit(train_network_edgelist)
        source_emb, target_emb = emb_object.model_return()

        predictions = sigmoid_predictor(source_emb.to_numpy(), 
                                        target_emb.to_numpy(), 
                                        remove_edge_info(test_recip_samples))
        recip_results = scores(true_recip_values, predictions)

        for_recip_json_file.update({f"{model_name}": recip_results})

        predictions = sigmoid_predictor(source_emb.to_numpy(), 
                                        target_emb.to_numpy(), 
                                        remove_edge_info(test_not_recip_samples))
        not_recip_results = scores(true_not_recip_values, predictions)

        for_not_recip_json_file.update({f"{model_name}": not_recip_results})

    # write json file with results
    with open(f"Results/Directed/Reciprocals/reciprocal_directed_test_procedure_results_run{run}_seed{seed}.json", "w", encoding="utf-8") as json_file:
        json.dump(for_recip_json_file, json_file, ensure_ascii=False, indent=4)

    with open(f"Results/Directed/NotReciprocals/not_reciprocal_directed_test_procedure_results_run{run}_seed{seed}.json", "w", encoding="utf-8") as json_file:
        json.dump(for_not_recip_json_file, json_file, ensure_ascii=False, indent=4)


def test_procedure(model_dict: dict, 
                   edgetuple_list: list, 
                   procedure: str = 'directed', 
                   emb_size: int = 16,  
                   runs: int = 1,
                   network_name = "Missing_name",
                   sampling_method: str = "uniform", 
                   split = [.8, .0, .2],
                   seed = 1234):
    
    seed_everything(seed)
    seeds = np.random.randint(0, 10**6, runs)
    
    num_layers = get_num_layers(edgetuple_list)

    if procedure == 'undirected':
        if not os.path.exists('Results/Undirected'):
            os.makedirs('Results/Undirected')

        for run, run_seed in enumerate(seeds):
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
                                                    split = split,
                                                    seed = new_seeds[target_layer - 1])
    
    elif procedure == 'directed':
        if not os.path.exists('Results/Directed/NotReciprocals'):
            os.makedirs('Results/Directed/NotReciprocals')
        if not os.path.exists('Results/Directed/Reciprocals'):
            os.makedirs('Results/Directed/Reciprocals')

        for run, run_seed in enumerate(seeds):
            seed_everything(run_seed)
            new_seeds = np.random.randint(0, 10**6, num_layers)

            for target_layer in range(1, num_layers + 1):
                directed_test_procedure_targetlayer(model_dict = model_dict, 
                                                    edgetuple_list = edgetuple_list, 
                                                    emb_size = emb_size,  
                                                    run = run,
                                                    target_layer = target_layer,
                                                    network_name = network_name,
                                                    sampling_method = sampling_method,
                                                    split = split, 
                                                    seed = new_seeds[target_layer - 1])
    
    else:
        return ValueError("Can only choose procedures: \'undirected\' or \'directed\'.")


if __name__ == "__main__":
    model_dict = {"liamne": liamne, "rmne": rmne, "mell": mell}

    dataframe = pd.read_csv("Vickers-Chan-7thGraders_multiplex.txt", sep=" ", header=None)

    data = list(dataframe.itertuples(index=False, name=None))

    test_procedure(model_dict, data, emb_size=EMB_SIZE, runs = 5, sampling_method="uniform", split = [.8, .0, .2],)

    print("fin")