import torch
import torch_geometric
from random import random
import networkx as nx

degree = torch_geometric.utils.degree
softmax = torch.nn.Softmax(dim=0)
def sample(weights, temperature=1):
	# samples randomly from a list of weights
	# probabilities of indices given by softmax(temperature * weights)
	seed = random()
	probabilities = softmax(temperature * weights)
	N = len(weights)
	for i in range(N):
		seed -= probabilities[i]
		if seed < 0:
			return i
	return N - 1

def balanced_forman(i, j, G):
	# Calculates Ric(i, j) for a graph G of type networkx.DiGraph
	di = G.degree(i)
	dj = G.degree(j)
	if di <= 1 or dj <= 1:
		return 0
	neighbors_of_i = list(G.neighbors(i))
	neighbors_of_j = list(G.neighbors(j))
	num_triangles = 0
	for v in neighbors_of_i:
		if v in neighbors_of_j:
			num_triangles += 1
	potential_squares = []
	neighbors_of_i_only = [v for v in neighbors_of_i if not v in neighbors_of_j and not v == j]
	neighbors_of_j_only = [v for v in neighbors_of_j if not v in neighbors_of_i and not v == i]
	for v in neighbors_of_i_only:
		for w in G.neighbors(v):
			if w in neighbors_of_j_only:
				potential_squares.append((v, w))
	squares_at_i = []
	squares_at_j = []
	for (v, w) in potential_squares:
		if not v in squares_at_i:
			squares_at_i.append(v)
		if not w in squares_at_j:
			squares_at_j.append(w)
	num_squares_i = len(squares_at_i)
	num_squares_j = len(squares_at_j)
	gamma_max = 0
	for k in squares_at_i:
		potential_gamma = 0
		for w in G.neighbors(k):
			if w in neighbors_of_j and not w in neighbors_of_i:
				potential_gamma += 1
		potential_gamma -= 1
		gamma_max = max(gamma_max, potential_gamma)
	for k in squares_at_j:
		potential_gamma = 0
		for w in G.neighbors(k):
			if w in neighbors_of_i and not w in neighbors_of_j:
				potential_gamma += 1
		potential_gamma -= 1
		gamma_max = max(gamma_max, potential_gamma)
	if gamma_max == 0:
		ric = 2/di + 2/dj - 2 + 2 * num_triangles / max(di, dj) + num_triangles / min(di, dj)
	else:
		ric = 2/di + 2/dj - 2 + 2 * num_triangles / max(di, dj) + num_triangles / min(di, dj) + (num_squares_i + num_squares_j)/(gamma_max * max(di, dj))
	return ric

def SDRF(x, edge_index, max_iterations=100, temperature=1):
	# stochastic discrete ricci flow
	num_nodes = len(x)
	G = torch_geometric.utils.to_networkx(num_nodes)
	# still have to work on this
def rlef(edge_index, edge_weights=None, num_iterations=100):
	# algorithm 1 from Overleaf (Random Local Edge Flip)
	if edge_weights == None:
		edge_weights = torch.ones(len(edge_index))
	for iteration in range(num_iterations):
		print(edge_index)
		input()
		index = sample(edge_weights, temperature=1)
		u = edge_index[0][index].item()
		v = edge_index[1][index].item()
		u_neighbors = edge_index[1,:][edge_index[0,:] == u]
		v_neighbors = edge_index[1,:][edge_index[0,:] == v]
		# update v to only include the neighbors that are eligible
		v_neighbors = torch.tensor([i.item() for i in v_neighbors if i != u and not i in u_neighbors])
		deg_u = len(u_neighbors)
		deg_v = len(v_neighbors)
		i = u_neighbors[sample(torch.ones(deg_u))]
		if i in v_neighbors or i == v or len(v_neighbors) == 0:
			pass
		else:
			j = v_neighbors[sample(torch.ones(deg_v))]
			for m in range(len(edge_index[0])):
				if edge_index[0][m] == i and edge_index[1][m] == u:
					edge_index[1][m] = v
				elif edge_index[0][m] == u and edge_index[1][m] == i:
					edge_index[0][m] = v
				elif edge_index[0][m] == j and edge_index[1][m] == v:
					edge_index[1][m] = u
				elif edge_index[0][m] == v and edge_index[1][m] == j:
					edge_index[0][m] = u
	return edge_index


edge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [1, 2, 0, 3, 0, 1]])
rlef(edge_index)