import os
import networkx as nx
import numpy as np
import gensim
import sys
import pathlib
import argparse
import pandas as pd
import tensorflow

import sklearn

from models import BasicModel

sys.path.clear
current_dir_path = str(pathlib.Path().resolve())

sys.path.clear()
sys.path.insert(0, current_dir_path + "/MELL")
sys.path.insert(0, current_dir_path)

###############################################################################################################
###################
#  MELL           #
###################

from MELL.MELL import MELL_model

"""
Comment on the model:
The MELL algorithm does not consider the weights of a weighted network. Nodes
must be given consecutive numbering starting from zero.
"""



class mell(BasicModel):
    '''
    MELL Model
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
        
        super(mell, self).__init__()
        self.name = "MELL"
        

        # parameters
        self.d     = params["emb_size"]
        self.k     = 4
        self.lamm  = 10
        self.beta  = 1
        self.gamma = 1
        self.eta   = 0.075
        self.max_iter = 500
        self.directed = params["directed"]
        self.dimension = params["emb_size"]
        self.weighted = params["weighted"]
        self.dir = "MELL_DATA/" + params["dataset"] +"_"+ str(params["run"])+"_"+ str(params["layer"])
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        
        

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
        self.L = len(list(dataset[0].unique()))
        nodes = list(set(dataset[1]).union(set(dataset[2])))
        self.N = len(nodes)
        
        #node-index dict
        self.node2idx = dict(zip(nodes,list(range(self.N))))
        
        def mymap(x):
            return self.node2idx.get(x)
        
        dataset[0] -= 1
        dataset[1] = dataset[1].map(mymap)
        dataset[2] = dataset[2].map(mymap)
        
        if self.weighted:
            dataset[3] = 1
        edges = dataset.values.tolist()
        self.M = len(edges)
        
        return edges
    
    def model_fit(self, edges):
        """
        Constructs network embedding from network data in form on dict list
        
        ...
        
        Parameters
        ----------
        data : list of lists
            

        Returns
        -------
        None.

        """
        
        model = MELL_model(self.L, self.N, self.directed, edges, self.d, self.k, self.lamm, self.beta, self.gamma, self.eta)
        model.train(self.max_iter)
        #model.save_embedding(self.dir)
        VH = model.resVH
        VT = model.resVT
        R = model.resR
        
        
                 
    
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

data = pd.read_csv("Datasets\Vickers\Vickers-Chan-7thGraders_multiplex.txt", sep=" ", header=None)

model = mell({"weighted": False, "directed": True, "emb_size": 16, "dataset": "Vickers", "run": 1, "layer":1})

model.fit(data)

print(model.model_return())

#print("fin")