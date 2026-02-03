import numpy as np
import pandas as pd
import torch
from collections import defaultdict

from models.abstract_model import BasicModel

###############################################################################################################
###################
# LIAMNE          #
###################

import LIAMNE.utils as LIAMNE_utils
import LIAMNE.model as LIAMNE_model
import LIAMNE.main as LIAMNE_main
from tqdm import tqdm
from collections import defaultdict



class liamne(BasicModel):
    '''
    LIAMNE Model
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
    _preprocessing(train_dataset)
        Preprocesses the input dataset into something that is an input to fit
    fit(dataset)
        In inherited method from BasicModel which calls on _preprocessing and _model_fit
    _model_fit(train_dataset)
        Preprocesses and fits the model
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
        
        super(liamne, self).__init__()
        self.name = "LIAMNE"
        self.model = LIAMNE_model.LIAMNE(params.get("node_num"), params.get("layer_num"), params.get("emb_size"), None, None)
        self.node_num = params.get("node_num") - 1
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
        train_data : dictionary of lists of nodes ordered by layer.
        '''
        dataset[1] -= 1
        dataset[2] -= 1
        node_num = 0
        data = dataset.to_numpy()
        data = data[:,:3]
        train_data = defaultdict(list)
        for l, i, j in data:
            train_data[l-1].append([int(i), int(j), 1])
            node_num = max(node_num, int(i), int(j))
        
        return train_data
    
    def _model_fit(self, data):
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
        nodes = list(range(self.node_num+1))
        train_data = data
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model.to(device)
    
        bce = torch.nn.BCEWithLogitsLoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
    
        for epoch in range(10):
            layer_embs = self.model.layer_embs.detach().cpu()
            new_network, new_neighs = LIAMNE_utils.under_sample(nodes, train_data, self.model.layer_num, self.target_layer-1, layer_embs, .2, .6)
            all_neighs = LIAMNE_utils.generate_neighs(new_neighs, 10, self.model.layer_num, self.node_num)
            tmp_neighs = torch.LongTensor(all_neighs)
    
            pairs = []
            for node, neighs in enumerate(all_neighs):
                for layer, neigh in enumerate(neighs):
                    temp = [node] + neigh
                    pairs.extend(LIAMNE_utils.generate_pairs(temp, 3, layer))
    
    
            batches = LIAMNE_main.get_batches(pairs, all_neighs, 128)
            data_iter = tqdm(
                batches,
                desc="epoch %d" % (epoch),
                total=(len(pairs) + (128 - 1)) // 128,
                bar_format="{l_bar}{r_bar}",
            )
    
            for i, pos_pairs in enumerate(data_iter):
                optim.zero_grad()
    
                final_emb_i = self.model(pos_pairs[0], pos_pairs[1], pos_pairs[3])
                final_emb_j = self.model(pos_pairs[0], pos_pairs[2], pos_pairs[4])
                
                score = torch.sum(final_emb_i*final_emb_j, dim=1)
                
                neg_pairs = LIAMNE_utils.gen_neg_pairs(self.node_num, self.model.layer_num-1, self.target_layer-1, 1, pos_pairs[1], new_neighs)
                neg_pairs = np.array(neg_pairs)
                
                final_emb_x = self.model(neg_pairs[:, 0], neg_pairs[:, 2], tmp_neighs[neg_pairs[:, 2]])
                neg_score = torch.sum(final_emb_i*final_emb_x, dim=1)
    
                labels = torch.ones(len(score)).to(device)
                neg_labels = torch.zeros(len(neg_score)).to(device)
    
                loss = bce(torch.cat((score, neg_score)), torch.cat((labels, neg_labels)))
    
                loss.backward()
                optim.step()
    
    
    
            layers_t = torch.LongTensor([self.target_layer-1 for _ in range(self.node_num+1)]).to(device)
            nodes_t = torch.LongTensor(list(range(self.node_num+1))).to(device)
            neighs_t = tmp_neighs[nodes_t]

            self.embeddings = self.model.forward(layers_t, nodes_t, neighs_t).cpu().detach()
            
    
    def model_return(self):
        """

        Returns
        -------
        W.T : dataframe
            reformats the computed embeddings to dataframe where each column represents the embedding vector on one node.

        """
        W = pd.DataFrame(self.embeddings.numpy())
        W.index += 1
        return W.T, W.T

