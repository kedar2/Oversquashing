import torch
import torch_geometric
import numpy as np
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

def spectral_gap(G):
	return nx.laplacian_spectrum(G)[1]

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
	neighbors_of_i_only = neighbors_of_i.difference(neighbors_of_j).difference({j})
	neighbors_of_j_only = neighbors_of_j.difference(neighbors_of_i).difference({i})
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
	return ric

def compute_curvature(G):
	# computes Ric(i, j) for all edges (i, j)
	curvatures = {}
	for edge in G.edges:
		(u, v) = edge
		curvatures[(u,v)] = balanced_forman(u, v, G)
	return curvatures

def sdrf(G, curvatures=None, max_iterations=100, temperature=5):
	# stochastic discrete ricci flow
	num_nodes = len(G.nodes)
	num_edges = len(G.edges)
	if curvatures == None:
		curvatures = compute_curvature(G)
	for iteration in range(max_iterations):
		(u, v) = argmin(curvatures)
		print(u, v, curvatures[(u,v)])
		ric_uv = curvatures[(u, v)]
		improvements = {}
		for k in G.neighbors(u):
			for l in G.neighbors(v):
				G_new = G.copy()
				a = min(k, l)
				b = max(k, l)
				G_new.add_edge(a, b)
				improvements[(a,b)] = balanced_forman(u, v, G_new) - ric_uv
		improvements_list = [[k, l, improvements[(k,l)]] for (k, l) in improvements]
		improvement_values = [x[2] for x in improvements_list]
		chosen_index = sample(improvement_values,temperature=temperature)
		i = improvements_list[chosen_index][0]
		j = improvements_list[chosen_index][1]
		G.add_edge(i, j)
		# need to update curvatures at neighbors of i and j
		edges_to_update = set()
		for w in G.neighbors(i):
			a = min(w, i)
			b = max(w, i)
			edges_to_update.add((a,b))
		for x in G.neighbors(j):
			a = min(x, j)
			b = max(x, j)
			edges_to_update.add((a,b))
		for w in G.neighbors(i):
			for x in G.neighbors(j):
				a = min(w, x)
				b = max(w, x)
				edges_to_update.add((a,b))
		for edge in edges_to_update:
			(w, x) = edge
			curvatures[(w, x)] = balanced_forman(w, x, G)
	return G, curvatures

def rlef(G):
	# algorithm 1 from Overleaf (Random Local Edge Flip)
	edge_list = list(G.edges)
	chosen_edge = edge_list[np.random.randint(len(edge_list))]
	(u, v) = chosen_edge
	i = np.random.choice(list(G.neighbors(u)))
	if  i in G.neighbors(v) or i == v:
		return G
	else:
		eligible_nodes = set(G.neighbors(v)).difference(set(G.neighbors(u))).difference({u})
		if eligible_nodes == set():
			return G
		else:
			j = np.random.choice(list(eligible_nodes))
			G.remove_edge(i,u)
			G.remove_edge(j,v)
			G.add_edge(i,v)
			G.add_edge(j,u)
		return G
