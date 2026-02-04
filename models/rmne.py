import torch
from torch.autograd import Variable
import os
import networkx as nx
import numpy as np
import random
import gc
import argparse
import pandas as pd
import itertools

from models.abstract_model import BasicModel


###############################################################################################################
###################
# RMNE            #
###################

from RMNE.main import RMNE, parameter_parser
from RMNE.utils import *
import RMNE.generate_pairs as generate_pairs
import RMNE.generate_roles as generate_roles
from RMNE.generate_roles import add_one_node


"""
Comment on the model:
The RMNE model is not implemented for directed networks, hence it will
only consider the underlying undirected network and in case of weighted 
directed networks the maximum of the weights is used as the undirected 
edge weight. The original model did not allow for more than 3 layers, hence
we needed to rewrite the function construct_role_pairs. Additionally, the function
get_parser was modified to directly return the default values.
"""

def construct_role_pairs(path, nodes_idx_nets, structure_feature_list, graphs_roles, nview, view_id):
    nodes_idx = nodes_idx_nets#[view_id]
    print("nodes_idx length: ")
    print(len(nodes_idx))
    # view_ids = [i for i in range(len(nodes_idx_nets))]
    role = []
    # if nview==3:
    for layer in range(nview): 
        role_pairs = []
        for node in nodes_idx:
            node = str(node)
            roleOFnode = structure_feature_list[view_id][node][0]
            
            role_pairs.append(add_one_node(graphs_roles,id=layer, roleOFnode=roleOFnode,node=node))  ##
            
        print(f"pairs {layer} length:  ")
        print(len(role_pairs))
        
        role.append(np.array(role_pairs))

    return role

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--read_pair',nargs='?',  default=True,
                        help='Default is true. If true, enables you to use the pairs you already have with your own deepwalk/node2vec setting')

    parser.add_argument('--input_graphs', nargs='?', default=r'./data/networks/',
                        help='Input graph path ')

    parser.add_argument('--input_pairs', nargs='?', default=r'./data/pairs/',
                        help='Input pairs path')

    parser.add_argument('--output', nargs='?', default=r'./output/emb/',
                        help='Embeddings path')

    parser.add_argument('--output_pairs', nargs='?', default=r'./output/pairs/',
                        help='Pairs output path')

    parser.add_argument('--dataset', nargs='?', default='LinkedIn/', #
                        help='Input graph path ')

    parser.add_argument('--nviews', type=int, default=3,
                        help='Number of views in dataset, i.e, if there are two networks nviews=3. Default is 3.')

    parser.add_argument('--dimensions', type=int, default=42,
                        help='Number of dimensions. Default is 128/|V|, i.e., 42 for 3 views, 128 for 1 view/network.')

    parser.add_argument('--alpha', type=float, default=1,
                        help='Hyperparameter for 1st order collaboration. Default is 1.')

    parser.add_argument('--beta', type=float, default=1,
                        help='Hyperparameter for 2nd order collaboration. Default is 1.')

    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Hyperparameter for 3nd order collaboration. Default is 1.')

    parser.add_argument('--walk_length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num_walks', type=int, default=5,
                        help='Number of walks per node. Default is 5.')

    parser.add_argument('--window_size', type=int, default=3,
                        help='Context size for optimization. Default is 3.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')

    parser.add_argument('--unweighted', dest='unweighted', action='store_false')

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')

    parser.add_argument('--undirected', dest='undirected', action='store_false')

    parser.add_argument('-lr', '--learning_rate',type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-ns', '--negative_sampling',type=float,
                        help='learning rate for the model, default=10',
                        default=10)

    parser.add_argument('-bs', '--batch_size',type=float,
                        help='batch size for the model, default=256',
                        default=256)

    parser.add_argument('-nepoch', '--epochs',
                        type=int,
                        help='number of training epochs',
                        default=10)
    parser.add_argument('--cuda',
                        action='store_false',
                        help='enables cuda. Default cuda.')
    return parser.parse_args(args=[])



class rmne(BasicModel):
    '''
    RMNE Model
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
        
        super(rmne, self).__init__()
        self.name = "RMNE"
        
        self.args = parameter_parser()
        self.params = get_parser()
        
        self.params.dimensions = params.get("emb_size")
        if params.get("weighted"):
            self.params.weighted = True
            self.params.unweighted = False
        else:
            self.params.weighted = False
            self.params.unweighted = True
        
        if params.get("directed"):
            self.params.directed = True
            self.params.undirected = False
        else:
            self.params.directed = False
            self.params.undirected = True
        
        self.params.dataset = params.get("dataset")
        
        self.newpath = r'./output/pairs/' + params.get("dataset") + "_" + str(params.get("seed"))
        if not os.path.exists(self.newpath):
            os.makedirs(self.newpath)
        
        

    def _preprocessing(self, dataset: list):
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
        dataset = sorted(dataset, key=lambda x: x[0])
        self.layers = dataset[-1][0]
        self.params.nviews = self.layers
        
        #because the rmne produces an embedding vector for each layer and concatenates them for the final embedding
        self.params.dimensions = self.params.dimensions//self.params.nviews
        
        def renumerate_node(x: tuple): # needs more nodes than layers
            return (x[0], x[1]-1, x[2]-1) 

        dataset = list(map(renumerate_node, dataset))
        
        data = list()
        self.missing_nodes = list()
        for layer in range(1, 1+ self.layers):
            layer_data = list()
            for edge in dataset:
                if edge[0] == layer:
                    layer_data.append((edge[1], edge[2]))
            data.append(layer_data)

        all_edges = list(itertools.chain(layer_data))

        nodes = list(set(itertools.chain.from_iterable(all_edges)))

        for layer in range(1, 1+ self.layers):
            layer_nodes = list(set(itertools.chain.from_iterable(layer_data)))
            missing_nodes = list(set(nodes).difference(set(layer_nodes)))
            self.missing_nodes.append(missing_nodes)

        return data
    
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
         
        if torch.cuda.is_available() and not self.params.cuda:
            print("WARNING: You have a CUDA device, you may try cuda with --cuda")
        device = 'cuda:0' if torch.cuda.is_available() and self.params.cuda else 'cpu'
        self.params.device = device
        print("Running on device: ", device)
        ####
        
        G = []
        for i in range(self.params.nviews):
            Graph = nx.from_edgelist(data[i]) ### changed to read edgelist instead of pandas adj
            Graph.add_nodes_from(self.missing_nodes[i])
            G.append(Graph)
            #print(111, sorted(G[-1].nodes), len(G[-1].nodes))
        
        structure_feature_list, graphs_roles = generate_roles.structual_roles(self.args, G)
        
        common_nodes = sorted(set(G[0]).intersection(*G))
        print('Number of common/core nodes in all networks: ', len(common_nodes))
        node2idx = {n: idx for (idx, n) in enumerate(common_nodes)}
        idx2node = {idx: n for (idx, n) in enumerate(common_nodes)}
    
        # relabeled_G = relabel_G(G, node2idx)
        
        nodes_idx_nets = []
        neigh_idx_nets = []
        node_role_nets = []
        print("nviews", self.params.nviews)
        for n_net in range(self.params.nviews):
            view_id = n_net + 1
            print("View ", view_id)

            nodes_idx, neigh_idx = generate_pairs.construct_word2vec_pairs(G[n_net], view_id, common_nodes, self.params.p,
                                                                           self.params.q, self.params.window_size,
                                                                           self.params.num_walks,
                                                                           self.params.walk_length,
                                                                           self.newpath,
                                                                           node2idx)
            
            
            nodes_idx_nets.append(nodes_idx)
            neigh_idx_nets.append(neigh_idx)
            """
            role_pairs_00, role_pairs_01, role_pairs_02 = generate_roles.construct_role_pairs(
                self.params.output_pairs + self.params.dataset, nodes_idx,  structure_feature_list, graphs_roles, self.params.nviews, n_net)
            
            node_role_nets.append([role_pairs_00, role_pairs_01, role_pairs_02])
            """
            node_role_nets.append(construct_role_pairs(
                self.newpath, nodes_idx,  structure_feature_list, graphs_roles, self.params.nviews, n_net))
            
        print(len(node_role_nets[0]))
        multinomial_nodes_idx = degree_nodes_common_nodes(G, common_nodes, node2idx)
    
        embed_freq = Variable(torch.Tensor(multinomial_nodes_idx))
        
        self.model = RMNE(self.params, len(common_nodes), embed_freq, self.params.batch_size)
        self.model.to(device)
    
        epo = 0
        min_pair_length = nodes_idx_nets[0].size
        for n_net in range(self.params.nviews):
            if min_pair_length > nodes_idx_nets[n_net].size:
                min_pair_length = nodes_idx_nets[n_net].size
        print("Total number of pairs: ", min_pair_length)
        print("Training started! \n")
        
    
        while epo <= self.params.epochs - 1:
    
            epo += 1
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
            running_loss = 0
            num_batches = 0
            shuffle_indices_nets = []
            #fifty = False
    
            for n_net in range(self.params.nviews):
                shuffle_indices = [x for x in range(nodes_idx_nets[n_net].size)]
                random.shuffle(shuffle_indices)
                shuffle_indices_nets.append(shuffle_indices)
            for count in range(0, min_pair_length, self.params.batch_size):
                optimizer.zero_grad()
                loss = self.model(count, shuffle_indices_nets, nodes_idx_nets, neigh_idx_nets, node_role_nets, self.params.alpha, self.params.beta, self.params.gamma)
                loss.backward()
                optimizer.step()
                running_loss += loss.detach().item()
                num_batches += 1
                torch.cuda.empty_cache()
                gc.collect()
    
        concat_tensors = self.model.node_embeddings[0].weight.detach().cpu()
    
        for i_tensor in range(1, self.model.num_net):
            concat_tensors = torch.cat((concat_tensors, self.model.node_embeddings[i_tensor].weight.detach().cpu()), 1)
    
        self.embs = np.array(concat_tensors)
        

    def model_return(self):
        """

        Returns
        -------
        W.T : dataframe
            reformats the computed embeddings to dataframe where each column represents the embedding vector on one node.

        """
        W = pd.DataFrame(self.embs)
        W.index += 1
        return W.T, W.T




