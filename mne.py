import os
import networkx as nx
import numpy as np
import gensim
import sys
import pathlib
import argparse
import pandas as pd
from gensim.models import word2vec

from models import BasicModel

sys.path.clear
current_dir_path = str(pathlib.Path().resolve())

sys.path.clear()
sys.path.insert(0, current_dir_path + "/MNE")
sys.path.insert(0, current_dir_path)

###############################################################################################################
###################
#  MNE            #
###################

from MNE import get_G_from_edges, MNE
from Random_walk import RWGraph

"""
Comment on the model:
gensim version 4 and above are not applicable because some key components have been changed in the code
"""

def train_embedding(current_embedding, walks, layer_id, dim, iter=10, info_size=10, base_weight=1):
    training_data = list()
    for walk in walks:
        tmp_walk = list()
        for node in walk:
            tmp_walk.append(str(node))
        training_data.append(tmp_walk)
    base_embedding = dict()
    if current_embedding is not None:
        for pos in range(len(current_embedding['index2word'])):
            base_embedding[current_embedding['index2word'][pos]] = current_embedding['base'][pos]
        if layer_id in current_embedding['tran']:
            current_tran = current_embedding['tran'][layer_id]
            current_additional_embedding = dict()
            for pos in range(len(current_embedding['index2word'])):
                current_additional_embedding[current_embedding['index2word'][pos]] = current_embedding['addition'][layer_id][pos]
            initial_embedding = {'base': base_embedding, 'tran': current_tran, 'addition': current_additional_embedding}
        else:
            initial_embedding = {'base': base_embedding, 'tran': None, 'addition': None}
    else:
        initial_embedding = None
    new_model = MNE(training_data, size=dim, window=5, min_count=0, sg=1, workers=4, iter=iter, small_size=info_size, initial_embedding=initial_embedding, base_weight=base_weight)
    # new_model = merge_model(tmp_model, new_model, w=learning_rate)

    return new_model.in_base, new_model.in_tran, new_model.in_local, new_model.wv.index2word

class mne(BasicModel):
    '''
    MNE Model
    ...
    
    Parameters
    ----------
    params : dict

    Attributes
    ----------
    Same as BasicModel class

    Methods
    -------
    Same as BasicModel class
    preprocessing(train_dataset)
        Preprocesses the input dataset into something that is an input to fit
    model_fit(train_dataset)
        Preprocesses and fits the model
    model_predict_proba(test_dataset)
        Outputs embedding matrix of the fitted model on test_dataset
    model_return()
        reformats the embedding matrix to a dataframe used in the pipeline
    '''
    def __init__(self, params):
        '''
        Creates an instance of liamne

        ...

        Parameters
        ----------
        params : dict
            
        '''
        
        super(mne, self).__init__()
        self.name = "MNE"
        self.directed = params["directed"]
        self.dimension = params["emb_size"]
        
        
        

    def preprocessing(self, dataset, is_training=True):
        '''
        Preprocessing step, which converts the network of dataframe format into dictionary list.

        ...

        Parameters
        ----------
        dataset : Dataset
            dataset to convert

        Returns
        ----------
        train_data : dictionary of lists of nodes ordered by layer.
        '''
        dataset = dataset.astype(str)
        nodes = list(set(dataset[1]).union(set(dataset[2])))
        edges = list(set(dataset[[1,2]].itertuples(index=False, name=None)))
        net_dict = dict()
        for layer in dataset[0].unique():
            data_ = dataset[dataset[0] == layer]
            lst = list(data_[[1,2]].itertuples(index=False, name=None))
            net_dict.update({layer : lst})
        net_dict['Base'] = edges
        return net_dict
    
    def model_fit(self, network_data):
        """
        Constructs network embedding from network data in form on dict list
        
        ...
        
        Parameters
        ----------
        data : dict list
            

        Returns
        -------
        None.

        """
        base_network = network_data['Base']
        base_G = RWGraph(get_G_from_edges(base_network), self.directed, 1, 1)
        print('finish building the graph')
        base_G.preprocess_transition_probs()
        base_walks = base_G.simulate_walks(20, 10)
        base_embedding, _, _, index2word = train_embedding(None, base_walks, 'Base', self.dimension, 100, 10, 1)
        final_model = dict()
        final_model['base'] = base_embedding
        final_model['tran'] = dict()
        final_model['addition'] = dict()
        final_model['index2word'] = index2word
        # you can repeat this process for multiple times
        for layer_id in network_data:
            if layer_id == 'Base':
                continue
            print('We are training model for layer:', layer_id)
            if layer_id not in final_model['addition']:
                final_model['addition'][layer_id] = np.zeros((len(final_model['index2word']), 10))
            tmp_data = network_data[layer_id]
            # start to do the random walk on a layer
            layer_G = RWGraph(get_G_from_edges(tmp_data), self.directed, 1, 1)
            layer_G.preprocess_transition_probs()
            layer_walks = layer_G.simulate_walks(20, 10)
            tmp_base, tmp_tran, tmp_local, tmp_index2word = train_embedding(final_model, layer_walks, layer_id, self.dimension, 20, 10, 0)
            base_embedding_dict = dict()
            local_embedding_dict = dict()
            for pos in range(len(tmp_index2word)):
                base_embedding_dict[tmp_index2word[pos]] = tmp_base[pos]
                local_embedding_dict[tmp_index2word[pos]] = tmp_local[pos]
            final_model['tran'][layer_id] = tmp_tran
            for tmp_word in tmp_index2word:
                final_model['addition'][layer_id][final_model['index2word'].index(tmp_word)] = local_embedding_dict[tmp_word]
        
        weights = {key:1 for key in final_model['addition']}

        indexes = [int(x) for x in final_model['index2word']]
        # Make sure that indexes start at 0
        if min(indexes) != 0:
            indexes = [x - min(indexes) for x in indexes]
        assert min(indexes) == 0
    
        base = final_model['base']
        assert base.shape[0] == len(indexes)
       
        embeddings = np.zeros(base.shape)
        for index in range(base.shape[0]):
            real_index = indexes[index]
            v_list = []
            for layer_id in final_model['addition']:
                # Equation 1 in paper
                v = base[index] + weights[layer_id] * np.dot(final_model['addition'][layer_id][index], final_model['tran'][layer_id])
                v_list.append(v)
            embeddings[real_index] = np.mean(v_list, axis=0)
        self.nodes = indexes
        self.embs = embeddings
        print("fin")
        
                 
    
    def model_return(self):
        """

        Returns
        -------
        W.T : dataframe
            reformats the computed embeddings to dataframe where each column represents the embedding vector on one node.

        """
        W = pd.DataFrame(self.embs, index=self.nodes)
        W.index += 1
        return W.T

"""
data = pd.read_csv("Datasets\Vickers\Vickers-Chan-7thGraders_multiplex.txt", sep=" ", header=None)

model = mne({"weighted": False, "directed": True, "emb_size": 16, "dataset": "Vickers/"})

model.fit(data)

print(model.model_return())

#print("fin")
"""