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
from GNN import GINLayer, MyGNN
import urllib3
urllib3.disable_warnings()

class Dataset(ABC):
    def __init__(self):
        self.data : Data
        self._init_dataset()

    def get_data(self):
        return self.data

    @abstractmethod
    def _init_dataset(self):
        pass

class LinkClassificationDataset(Dataset):
    def __init__(self):
        super().__init__()
        
    def _init_dataset(self):
        dataset = TUDataset('mutag', 'MUTAG')
        # Combining multiple graphs to have 1 big graph, but they are not connected
        all_edge_index = []
        all_x = []
        all_edge_attr = []
        node_offset = 0
        for data in dataset:
            all_x.append(data.x)
            all_edge_index.append(data.edge_index + node_offset)
            if data.edge_attr is not None:
                all_edge_attr.append(data.edge_attr.squeeze())
            node_offset += data.num_nodes
        combined_x = torch.cat(all_x, dim=0)
        combined_edge_index = torch.cat(all_edge_index, dim=1)
        combined_edge_attr = torch.cat(all_edge_attr, dim=0) if all_edge_attr else None
        self.data = Data(x=combined_x, edge_index=combined_edge_index, edge_attr=combined_edge_attr.argmax(-1))
    
class LinkPredictionDataset(Dataset):
    def __init__(self):
        super().__init__()
    
    def _init_dataset(self):
        dataset = Planetoid('cora', 'CORA')
        self.data = dataset[0]