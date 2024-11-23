from abc import ABC, abstractmethod
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from torch_geometric.utils import negative_sampling
import torch

class Element():
    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor, neg_edge_index: torch.Tensor):
        self.x = x
        self.edge_index = edge_index
        self.neg_edge_index = neg_edge_index

class Loader(ABC):
    def __init__(self, data):
        self.data = data
        self.train_data : Element
        self.val_data: Element
        self.test_data: Element
    

class LoaderTransductive(Loader):
    def __init__(self, data):
        super().__init__(data)
        train, val, test = RandomLinkSplit(split_labels=True)(self.data)
        self.train_data = Element(train.x, train.pos_edge_label_index, train.neg_edge_label_index)
        self.val_data = Element(val.x, val.pos_edge_label_index, val.neg_edge_label_index)
        self.test_data = Element(test.x, test.pos_edge_label_index, test.neg_edge_label_index)
    


class LoaderInductive(Loader):
    def __init__(self, data):
        super().__init__(data)
        train, val, test = self.__get_split()
        train_neg_edges = self.__get_negative_sampling(train)
        val_neg_edges = self.__get_negative_sampling(val)
        test_neg_edges = self.__get_negative_sampling(test)
        self.train_data = Element(train.x, train.edge_index, train_neg_edges)
        self.val_data = Element(val.x, val.edge_index, val_neg_edges)
        self.test_data = Element(test.x, test.edge_index, test_neg_edges)
    
    def __get_negative_sampling(self, data):
        return negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.x.size(0),
            num_neg_samples=data.edge_index.size(1)
        )

    def __get_split(self):
        split = RandomNodeSplit()
        data = split(self.data)
        train_data = self.data.subgraph(data.train_mask)
        val_data = self.data.subgraph(data.val_mask)
        test_data = self.data.subgraph(data.test_mask)
        return train_data, val_data, test_data