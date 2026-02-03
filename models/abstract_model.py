# BasicModel class is based on the work by 
# @article{reda2024stanscofi,
#  title={stanscofi and benchscofi: a new standard for drug repurposing by collaborative filtering},
#  author={R{\'e}da, Cl{\'e}mence and Vie, Jill-J{\^e}nn and Wolkenhauer, Olaf},
#  journal={Journal of Open Source Software},
#  volume={9},
#  number={93},
#  pages={5973},
#  year={2024}
# }
# https://github.com/RECeSS-EU-Project/
#

import numpy as np
import random
import pandas as pd
import torch
import sys
import os
import pathlib
#sys.path.clear
#current_dir_path = str(pathlib.Path().resolve())

###############################################################################################################
###################
# Basic model     #
###################

class BasicModel(object):
    '''
    Abstract class used to encode a multiplex network embedding

    ...

    Parameters
    ----------
    params : dict
        dictionary which contains method-wise parameters

    Attributes
    ----------
    name : str
        the name of the model
    model : depends on the implemented method
    ...
        other attributes might be present depending on the type of model

    Methods
    -------
    __init__(params)
        Constructor of class object
        Initializes the model with preselected parameters
    fit(train_dataset, seed=1234)
        Public class method which will be called upon from outside the class
        Preprocesses and fits the model 
    _preprocessing(train_dataset) [not implemented in BasicModel]
        Private class method only to be accessed from within the class
        Preprocess the input dataset into something that is an input to the self._model_fit if it exists
    _model_fit(train_dataset) [not implemented in BasicModel]
        Private class method only to be accessed from within the class
        Fits the model on train_dataset
    model_return() [not implemented in BasicModel]
        Public class method which will be called upon from outside the class
        Formats the embeddings to be returned
    '''
    def __init__(self):
        '''
        Creates an instance of BasicModel

        ...

        Parameters
        ----------
        params : dict
            dictionary which contains method-wise parameters
        '''
        self.name = "Model"
        

    def fit(self, train_dataset, seed=1234):
        '''
        Fitting the embedding model on the training dataset.

        Not implemented in the BasicModel class.

        ...

        Parameters
        ----------
        train_dataset : Dataset
            training dataset on which the model should fit
        seed : int (default: 1234)
            random seed
        '''
        np.random.seed(seed)
        random.seed(seed)
        self._model_fit(self._preprocessing(train_dataset))

    def _preprocessing(self, dataset):
        '''
        Preprocessing step, which converts elements of a dataset (multiplex networks and node attribute matrix) for the model 

        <Not implemented in the BasicModel class.>

        ...

        Parameters
        ----------
        dataset : 
            dataset to convert


        Returns
        ----------
        ... : ...
            appropriate inputs to the embedding model (varies across algorithms)
        '''
        raise NotImplemented

    def _model_fit(self):
        '''
        Fitting the model on the training dataset.

        <Not implemented in the BasicModel class.>

        ...

        Parameters
        ----------
        ... : ...
            appropriate inputs to the embedding model (varies across algorithms)
        '''
        raise NotImplemented
    
    def model_return(self):
        '''
        Format the embeddings to be returned
        
        <Not implemented in the BasicModel class.>

        ...
        
        Parameters
        ----------
        ... : ...
            appropriate inputs to the embedding model (varies across algorithms)
        '''
        raise NotImplemented
