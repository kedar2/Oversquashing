import torch
from torch import nn
from torch.nn import functional as F


class GraphModel(torch.nn.Module):
    def __init__(self, gnn_type, num_layers, dim0, h_dim, out_dim, last_layer_fully_adjacent,
                 unroll, layer_norm, use_activation, use_residual, num_nodes=0):
        super(GraphModel, self).__init__()
        self.gnn_type = gnn_type
        self.unroll = unroll
        self.last_layer_fully_adjacent = last_layer_fully_adjacent
        self.use_layer_norm = layer_norm
        self.use_activation = use_activation
        self.use_residual = use_residual
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.layers.append(gnn_type.get_layer(
                in_dim=dim0,
                out_dim=h_dim))
        if unroll:
            self.layers.append(gnn_type.get_layer(
                in_dim=h_dim,
                out_dim=h_dim))
        else:
            for i in range(num_layers):
                self.layers.append(gnn_type.get_layer(
                    in_dim=h_dim,
                    out_dim=h_dim))
        if self.use_layer_norm:
            for i in range(num_layers):
                self.layer_norms.append(nn.LayerNorm(h_dim))

        self.out_dim = out_dim
        self.out_layer = nn.Linear(in_features=h_dim, out_features=out_dim, bias=False)
        if self.last_layer_fully_adjacent:
            self.fully_adjacent_layer = nn.Linear(in_features=self.num_nodes, out_features=self.num_nodes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers):
            if self.unroll:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            new_x = x
            if self.last_layer_fully_adjacent and i == self.num_layers - 1:
                new_x = x + self.fully_adjacent_layer(x.T).T
            else:
                edges = edge_index
                new_x = layer(new_x, edges)
            if self.use_activation:
                new_x = F.relu(new_x)
            x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

        logits = self.out_layer(x)
        # logits = F.linear(root_nodes, self.layer0_values.weight)
        return logits
