import torch
import torch_geometric as pyg
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.datasets import Planetoid, Amazon, KarateClub, TUDataset
from torch_geometric.utils import train_test_split_edges, add_self_loops, negative_sampling
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
import torch.optim as optim
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import loaders
import importlib
importlib.reload(loaders)
from loaders import Loader, LoaderInductive, LoaderTransductive, Element
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from trainers import Trainer, LinkClassification, LinkPrediction

class GINLayer(gnn.MessagePassing):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float):
        super(GINLayer, self).__init__(aggr='add')
        self.mlp_f = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        self.mlp_o = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )
        self.eps = nn.Parameter(torch.rand(size=(1,)))
        self.skip = nn.Linear(hidden_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        out = self.propagate(edge_index=edge_index, x=x) # (num_nodes, hidden_features)
        skip_out = out
        eps = (1 + self.eps)
        out = eps * self.mlp_f(x) + out
        out = self.mlp_o(out) + self.skip(skip_out) # (num_nodes, out_features)
        out = self.dropout(out)
        out = self.norm(out)
        return out # (num_nodes, out_features)

    def message(self, x_j: torch.Tensor):
        return self.mlp_f(x_j) # (num_nodes, hidden_features)


class MyGNN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, num_layers: int, dropout: float):
        super(MyGNN, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList([])
        self.layers.append(GINLayer(in_features, hidden_features, hidden_features, self.dropout))
        for _ in range(num_layers - 1):
            self.layers.append(GINLayer(hidden_features, hidden_features, hidden_features, self.dropout))
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)        
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for layer in self.layers:
            x = layer(x, edge_index) # (num_nodes, hidden_features)
            # x = torch.relu(x)
        x_cat = self.__combine_node_embeddings(edge_index, x) # (num_nodes, 2 * hidden_features)
        x_cat = self.link_predictor(x_cat) # (num_nodes, out_features)
        return x_cat
    
    def __combine_node_embeddings(self, edges: torch.Tensor, nodes: torch.Tensor):
        return torch.cat([nodes[edges[0]], nodes[edges[1]]], dim=-1) # (num_nodes, 2 * hidden_features)