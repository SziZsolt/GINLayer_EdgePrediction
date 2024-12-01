from abc import ABC, abstractmethod
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from torch_geometric.utils import negative_sampling
import torch

class Element():
    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor, neg_edge_index: torch.Tensor, edge_classes: torch.Tensor):
        self.x = x
        self.edge_index = edge_index
        self.neg_edge_index = neg_edge_index
        self.edge_classes = edge_classes

class Loader(ABC):
    def __init__(self, data, edge_classes: torch.Tensor | None):
        self.data = data
        self.train_data : Element
        self.val_data: Element
        self.test_data: Element
        if edge_classes is not None:
            self.data.edge_classes = edge_classes
        else:
            self.data.edge_classes = torch.tensor([])

class LoaderTransductive(Loader):
    def __init__(self, data, edge_classes: torch.Tensor | None):
        super().__init__(data, edge_classes)
        train, val, test = RandomLinkSplit(split_labels=True, num_test=0.1, num_val=0.1)(self.data)
        self.train_data = Element(train.x, train.pos_edge_label_index, train.neg_edge_label_index,
                                  self.get_indices(self.data.edge_index, train.pos_edge_label_index, self.data.edge_classes))
        self.val_data = Element(val.x, val.pos_edge_label_index, val.neg_edge_label_index,
                                self.get_indices(self.data.edge_index, val.pos_edge_label_index, self.data.edge_classes))
        self.test_data = Element(test.x, test.pos_edge_label_index, test.neg_edge_label_index,
                                 self.get_indices(self.data.edge_index, test.pos_edge_label_index, self.data.edge_classes))
    
    def get_indices(self, edge_index: torch.Tensor, pos_edge_index: torch.Tensor, edge_classes: torch.Tensor):
        matches = (edge_index.T[:, None] == pos_edge_index.T).all(dim=2).any(dim=1)
        indices = torch.nonzero(matches).squeeze()
        return edge_classes[indices]

class LoaderInductive(Loader):
    def __init__(self, data, edge_classes: torch.Tensor | None):
        super().__init__(data, edge_classes)
        train, val, test = self.__get_split()
        train_neg_edges = self.__get_negative_sampling(train)
        val_neg_edges = self.__get_negative_sampling(val)
        test_neg_edges = self.__get_negative_sampling(test)
        self.train_data = Element(train.x, train.edge_index, train_neg_edges, train.edge_classes)
        self.val_data = Element(val.x, val.edge_index, val_neg_edges, val.edge_classes)
        self.test_data = Element(test.x, test.edge_index, test_neg_edges, test.edge_classes)
    
    def __get_negative_sampling(self, data):
        return negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.x.size(0),
            num_neg_samples=data.edge_index.size(1)
        )

    def __get_split(self):
        num_nodes = self.data.num_nodes
        train_ratio = 0.5
        val_ratio = 0.2
        indices = torch.randperm(num_nodes)
        train_idx = indices[:int(train_ratio * num_nodes)]
        val_idx = indices[int(train_ratio * num_nodes):int((train_ratio + val_ratio) * num_nodes)]
        test_idx = indices[int((train_ratio + val_ratio) * num_nodes):]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[val_idx] = True
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_idx] = True
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask
        split = RandomNodeSplit()
        data = split(self.data)
        train_data = self.data.subgraph(data.train_mask)
        val_data = self.data.subgraph(data.val_mask)
        test_data = self.data.subgraph(data.test_mask)
        return train_data, val_data, test_data