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

class Trainer(ABC):
    def __init__(self, model: nn.Module, optimizer, loader: Loader):
        self.optimizer = optimizer
        self.loader = loader
        self.model = model
        self.train_loss_hist = []
        self.val_loss_hist = []
        self.val_acc_hist = []

    @abstractmethod
    def forward(self, element: Element):
        pass

    @abstractmethod
    def get_loss(self, element: Element) -> torch.Tensor:
        pass

    @abstractmethod
    def get_acc(self, element: Element) -> torch.Tensor:
        pass

    def train(self, num_epochs: int, patience: int):
        patience_counter = 0
        best_model_wts = None
        best_loss = torch.inf
        self.train_loss_hist = []
        self.val_loss_hist = []
        self.val_acc_hist = []
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            loss = self.get_loss(self.loader.train_data)
            loss.backward()
            train_loss = loss.item()
            self.train_loss_hist.append(train_loss)
            self.optimizer.step()
            self.model.eval()
            with torch.inference_mode():
                val_loss = self.get_loss(self.loader.val_data).item()
                val_acc = self.get_acc(self.loader.val_data).item()
                self.val_loss_hist.append(val_loss)
                self.val_acc_hist.append(val_acc)
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_model_wts = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print('Early stop')
                        break
            print(f'Epoch: {epoch + 1} | Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val accuracy: {val_acc:.3f}')
        self.model.load_state_dict(best_model_wts)

    def plot_train_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(16,8))
        axes[0].plot(self.train_loss_hist, label='Train loss')
        axes[0].plot(self.val_loss_hist, label='Validation loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Losses')
        axes[0].grid()
        axes[1].plot(self.val_acc_hist, label='Validation accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].set_title('Accuracy')
        axes[1].grid()
        fig.suptitle('Metrics during training')
        plt.show()

    @abstractmethod
    def evaluate_test_set(self):
        pass


class LinkClassification(Trainer):
    def __init__(self, model, optimizer, loader):
        super().__init__(model, optimizer, loader)
        self.loss_fn = nn.CrossEntropyLoss()

    def __to_cat(self, arr: torch.Tensor):
        return arr.argmax(-1)

    def get_loss(self, element: Element):
        pred = self.forward(element)
        loss = self.loss_fn(pred, element.edge_classes)
        return loss
    
    def get_acc(self, element: Element):
        pred = self.forward(element)
        pred = torch.argmax(pred, dim=-1)
        return torch.sum(pred == element.edge_classes) / pred.shape[0]
    
    def forward(self, element: Element):
        return self.model(element.x, element.edge_index).squeeze()
    
    def evaluate_test_set(self):
        pred = self.forward(self.loader.test_data)
        acc = self.get_acc(self.loader.test_data)
        pred = self.__to_cat(pred)
        sns.heatmap(confusion_matrix(self.loader.test_data.edge_classes, pred), cmap='Blues', annot=True)
        plt.xlabel('Prediction')
        plt.ylabel('True')
        plt.title(f'Accuracy: {acc:.2f}')
        plt.show()

class LinkPrediction(Trainer):
    def __init__(self, model, optimizer, loader):
        super().__init__(model, optimizer, loader)

    def __to_cat(self, arr: torch.Tensor):
        return torch.round(torch.sigmoid(arr))
    
    def get_loss(self, element: Element):
        pos_scores, neg_scores = self.forward(element)
        pos_scores = torch.sigmoid(pos_scores)
        neg_scores = torch.sigmoid(neg_scores)
        return -torch.mean(torch.log(pos_scores) + torch.log(1 - neg_scores))
    
    def get_acc(self, element: Element):
        pos_scores, neg_scores = self.forward(element)
        pos_scores = self.__to_cat(pos_scores)
        neg_scores = self.__to_cat(neg_scores)
        n = pos_scores.shape[0] + neg_scores.shape[0]
        return (torch.sum(pos_scores == 1) + torch.sum(neg_scores == 0)) / n
    
    def forward(self, element: Element):
        pos_scores = self.model(element.x, element.edge_index)
        neg_scores = self.model(element.x, element.neg_edge_index)
        return pos_scores, neg_scores
    
    def evaluate_test_set(self):
        test_pos_scores, test_neg_scores = self.forward(self.loader.test_data)
        acc = self.get_acc(self.loader.test_data)
        test_pos_scores, test_neg_scores = test_pos_scores.detach(), test_neg_scores.detach()
        test_pos_scores = self.__to_cat(test_pos_scores)
        test_neg_scores = self.__to_cat(test_neg_scores)
        labels_pos = torch.ones_like(test_pos_scores)
        labels_neg = torch.zeros_like(test_neg_scores)
        labels = torch.cat([labels_pos, labels_neg])
        logits = torch.cat([test_pos_scores, test_neg_scores])
        sns.heatmap(confusion_matrix(labels, logits), annot=True, cmap='Blues')
        plt.xticks([0.5,1.5], ['Negative', 'Positive'])
        plt.yticks([0.5,1.5], ['Negative', 'Positive'])
        plt.xlabel('Prediction')
        plt.ylabel('True')
        plt.title(f'Accuracy: {acc:.2f}')
        plt.show()