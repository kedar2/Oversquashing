import torch
import numpy as np
from torch import nn
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.data import Data
from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import homophily, to_undirected
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_undirected, remove_self_loops
from torch.optim import Adam
from sklearn.model_selection import train_test_split
device = torch.device('cuda')

def determine_training_data(total_size, train_ratio):
	# generates a random vector of 0s and 1s with length total_size, with the fraction of 1s given by train_ratio
	rand_vector = torch.rand(total_size)
	k = int(train_ratio * total_size)
	k_th_quant = torch.topk(rand_vector, k, largest=False)

# Message-passing neural network with ReLU activation

class MPNN_Layer(MessagePassing):
	def __init__(self, in_channels, out_channels):
		super().__init__(aggr='add')
		self.mapping = Sequential(Linear(in_channels, out_channels),
		ReLU())
	def forward(self, x, edge_index):
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = self.mapping(x)
		# normalization
		row, col = edge_index
		deg = degree(col, x.size(0), dtype=x.dtype)
		deg_inv_sqrt = deg.pow(-0.5)
		deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
		norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

		return self.propagate(edge_index, x=x, norm=norm)
	def message(self, x_j, norm):
		return norm.view(-1, 1) * x_j

class MPNN(nn.Module):
	def __init__(self, num_layers, in_channels, hidden_channels, out_channels):
		super(MPNN, self).__init__()
		self.num_layers = num_layers
		self.in_channels = in_channels
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.layers = []
		for i in range(num_layers):
			if i == 0:
				self.layers.append(MPNN_Layer(in_channels=in_channels, out_channels=hidden_channels))
			elif i == num_layers - 1:
				self.layers.append(MPNN_Layer(in_channels=hidden_channels, out_channels=out_channels))
			else:
				self.layers.append(MPNN_Layer(in_channels=hidden_channels, out_channels=hidden_channels))
		self.layers = torch.nn.ModuleList(self.layers)
	def forward(self, x, edge_index):
		for layer in self.layers:
			x = layer(x=x, edge_index=edge_index)
		return x
# handle datasets

cornell = WebKB(root="data", name="Cornell")[0]
wisconsin = WebKB(root="data", name="Wisconsin")[0]
texas = WebKB(root="data", name="Texas")[0]
chameleon = WikipediaNetwork(root="data", name="chameleon")[0]
squirrel = WikipediaNetwork(root="data", name="squirrel")[0]
actor = Actor(root="data")[0]
cora = Planetoid(root="data", name="cora")[0]
citeseer = Planetoid(root="data", name="citeseer")[0]
pubmed = Planetoid(root="data", name="pubmed")[0]

largest_cc = LargestConnectedComponents()

graphs = [cornell, texas, wisconsin, chameleon, squirrel, actor, cora, citeseer, pubmed]
graph_names = ["Cornell", "Texas", "Wisconsin", "Chameleon", "Squirrel", "Actor", "Cora", "Citeseer", "Pubmed"]

for graph in graphs:
	graph.edge_index = to_undirected(graph.edge_index)
	graph = largest_cc(graph)
# calculate homophily of the graphs

#homophilies = []
#for graph in graphs:
#	undirected = remove_self_loops(to_undirected(graph.edge_index, num_nodes=graph.num_nodes))[0]
#	print(undirected.size(1)/2)
#	homophilies.append(homophily(edge_index=undirected, y=graph.y, method="node"))

graph_to_use = cora

net = MPNN(3, graph_to_use.x.size()[1], 64, max(graph_to_use.y) + 1)

# training

def train(x, edge_index, y, model, train_size=0.6, validation_size=0.2, num_iterations=1000):
	loss_fn = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters())
	test_size = 1 - train_size - validation_size
	indices = list(range(len(y)))
	non_test, test = train_test_split(indices, test_size=test_size)
	train, validation = train_test_split(non_test, test_size=validation_size/(validation_size + train_size))
	for i in range(num_iterations):
		optimizer.zero_grad()
		outputs = model(x, edge_index)
		train_loss = loss_fn(outputs[train], y[train])
		train_loss.backward()
		optimizer.step()
		predictions = torch.argmax(outputs[validation], dim=1)
		train_predictions = torch.argmax(outputs[train], dim=1)
		print("TRAIN ACCURACY: ",sum(train_predictions == y[train]).item() / len(y[train]))
		print("VALIDATION ACCURACY: ", sum(predictions == y[validation]).item() / len(y[validation]))
		
	
train(graph_to_use.x, graph_to_use.edge_index, graph_to_use.y, net)
