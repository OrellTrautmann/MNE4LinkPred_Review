import numpy as np
import pandas as pd
import itertools

from models.abstract_model import BasicModel

###############################################################################################################
###################
#  MELL           #
###################

from MELL.MELL.MELL import MELL_model

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
        self.d     = params.get("emb_size")
        self.k     = 4
        self.lamm  = 10
        self.beta  = 1
        self.gamma = 1
        self.eta   = 0.075
        self.max_iter = 500
        self.directed = params.get("directed")
        self.dimension = params.get("emb_size")
        self.weighted = params.get("weighted")
        self.target_layer = params.get("target_layer")
        

    def _preprocessing(self, dataset):
        '''
        Preprocessing step, which converts the network of dataframe format into dictionary list.

        ...

        Parameters
        ----------
        dataset : Dataset
            dataset to convert

        Returns
        ----------
        edges : dictionary of lists of nodes ordered by layer.
        '''
        dataset = sorted(dataset, key=lambda x: x[0])
        self.L = dataset[-1][0]
        nodes = list(set(itertools.chain.from_iterable(dataset)))
        self.N = len(nodes)
        
        #node-index dict
        self.node2idx = dict(zip(nodes,list(range(self.N))))
        
        def mymap(x):
            return self.node2idx.get(x)
        
        edges = [[edge[0]-1, mymap(edge[1]), mymap(edge[2]), 1] for edge in dataset]
        self.M = len(edges)
        
        return edges
    
    def _model_fit(self, edges):
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
        self.VH = model.resVH
        self.VT = model.resVT
        self.R = model.resR
    
    def model_return(self):
        """

        Returns
        -------
        W.T : dataframe
            reformats the computed embeddings to dataframe where each column represents the embedding vector on one node.

        """
        source_emb = pd.DataFrame(self.R[self.L-1] + self.VH[self.L-1,:,:])
        source_emb.index += 1
        target_emb = pd.DataFrame(self.VT[self.L-1,:,:])
        target_emb.index += 1
        return source_emb.T, target_emb.T
    
