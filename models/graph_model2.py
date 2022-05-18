__author__ = "Stefan Weißenberger and Johannes Gasteiger"
__license__ = "MIT"

from typing import List

import torch
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, InMemoryDataset


class GCN(torch.nn.Module):
    def __init__(self,
                 data,
                 hidden: List[int] = [64],
                 dropout: float = 0.5,
                 last_layer_fully_adjacent=False,
                 skip_connection=0):
        super(GCN, self).__init__()
        self.last_layer_fully_adjacent = last_layer_fully_adjacent
        num_labels = max(data.y).item() + 1
        num_features = [data.x.shape[1]] + hidden + [num_labels]
        num_nodes = data.x.shape[0]
        num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            if i == num_layers -1 and self.last_layer_fully_adjacent:
                layers.append(torch.nn.Linear(hidden[-1], num_labels))
                layers.append(torch.nn.Linear(num_nodes, num_nodes))
            else:
                layers.append(GCNConv(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()


    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        for i, layer in enumerate(self.layers):
            
            if self.last_layer_fully_adjacent and i == len(self.layers) - 2:
                    x = layer(x)
                    x = self.layers[-1](x.T).T
                    break
            x = layer(x, edge_index)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return torch.nn.functional.log_softmax(x, dim=1)
