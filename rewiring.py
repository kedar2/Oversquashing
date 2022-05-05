import torch
import torch_geometric
from numpy.random import random
import networkx as nx
from math import inf

degree = torch_geometric.utils.degree
softmax = torch.nn.Softmax(dim=0)

def argmin(d):
	smallest = inf
	for i in d:
		if d[i] < smallest:
			smallest = d[i]
			key_of_smallest = i
	return key_of_smallest

def sample(weights, temperature=1, use_softmax=True):
	# samples randomly from a list of weights
	weights = torch.tensor(weights)
	seed = random()
	if use_softmax:
		probabilities = softmax(temperature * weights)
	else:
		probabilities = weights / sum(weights)
	N = len(weights)
	for i in range(N):
		seed -= probabilities[i]
		if seed < 0:
			return i
	return N - 1

def second_neighborhood(i, G):
	# returns all vertices of distance at most 2 from a given vertex i in G
	second_neighbors = set()
	for j in G.neighbors(i):
		second_neighbors.add(j)
		for k in G.neighbors(j):
			second_neighbors = second_neighbors.union(neighbors_of_j)
	second_neighbors.add(i)
	return second_neighbors

def balanced_forman(i, j, G):
	# Calculates Ric(i, j) for a graph G of type networkx.DiGraph
	di = G.degree(i)
	dj = G.degree(j)
	if di <= 1 or dj <= 1:
		return 0
	neighbors_of_i = set(G.neighbors(i))
	neighbors_of_j = set(G.neighbors(j))
	num_triangles = 0
	triangles = neighbors_of_i.intersection(neighbors_of_j)
	num_triangles = len(triangles)
	potential_squares = set()
	neighbors_of_i_only = neighbors_of_i.difference(neighbors_of_j)
	neighbors_of_j_only = neighbors_of_i.difference(neighbors_of_j)
	for v in neighbors_of_i_only:
		for w in G.neighbors(v):
			if w in neighbors_of_j_only:
				potential_squares.add((v, w))
	squares_at_i = {v for (v, w) in potential_squares}
	squares_at_j = {w for (v, w) in potential_squares}
	for (v, w) in potential_squares:
		squares_at_i.add(v)
		squares_at_j.add(w)
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
	triangle_term = 2 * num_triangles / max(di, dj) + num_triangles / min(di, dj)
	if gamma_max == 0:
		square_term = 0
	else:
		square_term = (num_squares_i + num_squares_j)/(gamma_max * max(di, dj))
	ric = 2/di + 2/dj - 2 + triangle_term + square_term
	print(ric, triangle_term, square_term)
	return ric

def SDRF(G, max_iterations=100, temperature=1):
	# stochastic discrete ricci flow
	num_nodes = len(G.nodes)
	num_edges = len(G.edges)
	curvatures = {}
	for edge in G.edges:
		(u, v) = edge
		curvatures[(u,v)] = balanced_forman(u, v, G)
	for iteration in range(max_iterations):
		(u, v) = argmin(curvatures)
		ric_uv = curvatures[(u, v)]
		improvements = {}
		for k in G.neighbors(u):
			for l in G.neighbors(v):
				G_new = G.copy()
				G_new.add_edge(k, l)
				improvements[(k,l)] = balanced_forman(u, v, G_new) - ric_uv
		improvements_list = [[k, l, improvements[(k,l)]] for (k, l) in improvements]
		improvement_values = torch.tensor([x[2] for x in improvements_list])
		chosen_index = sample(improvement_values,temperature=temperature)
		i = improvements_list[chosen_index][0]
		j = improvements_list[chosen_index][1]
		G.add_edge(i, j)
	return G

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


#edge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [1, 2, 0, 3, 0, 1]])
#rlef(edge_index)
