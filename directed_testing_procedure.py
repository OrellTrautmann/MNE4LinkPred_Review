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
                   get_reciprocals, 
                   sample_not_reciprocals, 
                   find_reciprocals,
                   get_num_layers)

import random
import numpy as np
from tqdm import tqdm
import json


# import model classes
from models.mell import mell
from models.liamne import liamne
from models.rmne import rmne

EMB_SIZE = 16

def undirected_test_procedure(model_dict: dict, 
                              edgetuple_list: list, 
                              target_layer: int = 1, 
                              emb_size: int = 16,  
                              run: int = 1,
                              network_name = "Vickers",
                              sampling_method: str = "uniform", 
                              seed = 124):
    
    # generate reproducible random seeds 
    seed_everything(seed)
    seeds = np.random.randint(0, 10**6, 4)

    # construct training network
    target_edgetuple_list, aux_edgetuple_list = split_target_auxiliary_layers(edgetuple_list, 
                                                                              target_layer)
    train_edge_list, val_edge_list, test_edge_list = split_dataset(remove_edge_info(target_edgetuple_list), 
                                                                   split=[.8, .0, .2])
    train_network_edgelist = partial_multiplex(train_edge_list, 
                                               aux_edgetuple_list, 
                                               target_layer)

    # generate testing set with negative samples
    node_list = get_node_list(edgetuple_list)
    neg_samples = neg_sampling(target_edgetuple_list, 
                               node_list, 
                               len(test_edge_list), 
                               sampling_method=sampling_method)
    test_samples = add_edge_info(test_edge_list, 
                                 target_layer, 
                                 weight = 1) 
    test_samples += neg_samples
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
                                             "run": run})
        
        emb_object.fit(train_network_edgelist)
        source_emb, target_emb = emb_object.model_return()

        predictions = sigmoid_predictor(source_emb.to_numpy(), 
                                        target_emb.to_numpy(), 
                                        remove_edge_info(test_samples))
        results = scores(true_values, predictions)

        for_json_file.update({f"{model_name}": results})

    # write json file with results
    with open(f"undirected_test_procedure_results_run{run}_seed{seed}.json", "w", encoding="utf-8") as json_file:
        json.dump(for_json_file, json_file, ensure_ascii=False, indent=4)



def un_vs_directed_test_procedure(model_dict, edgetuple_list: list, target_layer: int, sampling_method: str = "uniform", seed = 124):
    seed_everything(seed)
    node_list = get_node_list(edgetuple_list)
    target_edgetuple_list, aux_edgetuple_list = split_target_auxiliary_layers(edgetuple_list, target_layer)
    train_edge_list, val_edge_list, test_edge_list = split_dataset(remove_edge_info(target_edgetuple_list), split=[.8, .0, .2])
    train_network_edgelist = partial_multiplex(train_edge_list, aux_edgetuple_list, target_layer)
    
    # undirected part
    neg_samples = neg_sampling(target_edgetuple_list, node_list, len(test_edge_list), sampling_method=sampling_method)
    test_samples = add_edge_info(test_edge_list, target_layer, 1) + neg_samples
    random.shuffle(test_samples)
    true_values = [edge[3] for edge in test_samples]

    sample_info = f"Number of negative samples: {len(neg_samples)}, number of reciprocals: {find_reciprocals(remove_edge_info(target_edgetuple_list), remove_edge_info(neg_samples))}, layer density: {len(target_edgetuple_list)/(len(node_list) * (len(node_list) - 1))}"

    with open("undirected_test_procedure_results.txt", "a") as file:
        file.write(sample_info + "\n")

    # directed part
    reciprocal_samples = get_reciprocals(target_edgetuple_list)
    not_reciprocal_samples = sample_not_reciprocals(edgetuple_list, node_list, len(test_edge_list))
    test_recip_samples = add_edge_info(test_edge_list, target_layer, 1) + reciprocal_samples
    test_not_recip_samples = add_edge_info(test_edge_list, target_layer, 1) + add_edge_info(not_reciprocal_samples, target_layer, 0)
    random.shuffle(test_recip_samples)
    true_recip_values = [edge[3] for edge in test_recip_samples]

    with open("directed_test_procedure_results.txt", "a") as file:
            file.write(f"Number of reciprocal samples: {len(reciprocal_samples)}, number of reciprocals: {find_reciprocals(remove_edge_info(target_edgetuple_list), remove_edge_info(reciprocal_samples))}, layer density: {len(target_edgetuple_list)/(len(node_list) * (len(node_list) - 1))}\n")

    random.shuffle(test_not_recip_samples)
    true_not_recip_values = [edge[3] for edge in test_not_recip_samples]

    with open("directed_test_procedure_results.txt", "a") as file:
        file.write(f"Number of non reciprocal samples: {len(not_reciprocal_samples)}, number of reciprocals: {find_reciprocals(remove_edge_info(target_edgetuple_list), remove_edge_info(not_reciprocal_samples))}, layer density: {len(target_edgetuple_list)/(len(node_list) * (len(node_list) - 1))}\n")


    for model_name in tqdm(model_dict):
        emb_object = model_dict[model_name]({"weighted": False, "directed": True, "emb_size": EMB_SIZE, "node_num": len(node_list), 
                                             "layer_num": 3, "dataset": "Name", "run": 1, "target_layer":1})
        
        emb_object.fit(train_network_edgelist)

        source_emb, target_emb = emb_object.model_return()

        # undirected part
        print(f"num elem in tuple: {test_samples[0]}")
        predictions = sigmoid_predictor(source_emb.to_numpy(), target_emb.to_numpy(), remove_edge_info(test_samples))
        results = scores(true_values, predictions)

        with open("undirected_test_procedure_results.txt", "a") as file:
            model_results = f"model: {model_name}, results: {results}"
            file.write(model_results + "\n")

        # directed part
        predictions = sigmoid_predictor(source_emb.to_numpy(), target_emb.to_numpy(), test_recip_samples)
        results = scores(true_recip_values, predictions)

        with open("directed_test_procedure_results.txt", "a") as file:
            file.write("{case: reciprocals, model: " + f"{model_name}" + ", results: " + f"{results}" + "}\n")

        predictions = sigmoid_predictor(source_emb.to_numpy(), target_emb.to_numpy(), test_not_recip_samples)
        results = scores(true_not_recip_values, predictions)

        with open("directed_test_procedure_results.txt", "a") as file:
            file.write("{case: not reciprocals, model: " + f"{model_name}" + ", results: " + f"{results}" + "}\n")




if __name__ == "__main__":
    model_dict = {"liamne": liamne, "rmne": rmne, "mell": mell}

    dataframe = pd.read_csv("Vickers-Chan-7thGraders_multiplex.txt", sep=" ", header=None)

    data = list(dataframe.itertuples(index=False, name=None))

    undirected_test_procedure(model_dict, data, target_layer=1, sampling_method="uniform")

    with open(f"undirected_test_procedure_results_run{1}_seed{124}.json", "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    print("data", data)

    print("fin")