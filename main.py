import torch
import numpy as np
import networkx as nx
from torch import nn
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.data import Data
from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import homophily, to_undirected, to_networkx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_undirected, remove_self_loops
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import rewiring
device = torch.device('cuda')

def determine_training_data(total_size, train_ratio):
	# generates a random vector of 0s and 1s with length total_size, with the fraction of 1s given by train_ratio
	rand_vector = torch.rand(total_size)
	k = int(train_ratio * total_size)
	k_th_quant = torch.topk(rand_vector, k, largest=False)

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

graph_to_use = wisconsin
G = to_networkx(cora, to_undirected=True)
#x = rewiring.compute_curvature(G)
#for i in range(10000000):
#	G = rewiring.rlef(G)
#	if i % 10000 == 0:
#		curvatures = rewiring.compute_curvature(G)
#		total_curvature = 0
#		for j in curvatures:
#			total_curvature += curvatures[j]
#		print("TOTAL CURVATURE: ", total_curvature)
#		print("SPECTRAL GAP: ", rewiring.spectral_gap(G))