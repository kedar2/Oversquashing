from preprocessing import rewiring
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import inf, log

def path_of_cliques(num_cliques, size_of_clique):
	G = nx.Graph([])
	for i in range(num_cliques):
		for j in range(size_of_clique):
			for k in range(j):
				G.add_edge(i*size_of_clique+j, i*size_of_clique+k)
		if i != num_cliques - 1:
			G.add_edge((i+1)*size_of_clique - 1, (i+1)*size_of_clique)
	return G

def dumbbell_graph(n, dumbbell_length):
	l = []
	for i in range(n):
		l.append((i, 2 * n - 1 + dumbbell_length + 2))
		for j in range(i):
			l.append((i, j))
	for i in range(n, 2*n):
		l.append((i, 2*n))
		for j in range(n, i):
			l.append((i,j))
	for i in range(2 * n, 2 * n - 1 + dumbbell_length + 2):
		l.append((i, i + 1))
	G = nx.Graph(l)
	return G


def ring_of_cliques(n, d):
	# ring of cliques graph with n vertices of degree d
	G = nx.Graph([])
	k = d + 1
	# encodes vertex by which clique it's in
	f = lambda x: (x % k, x // k)
	for x1 in range(n):
		for x2 in range(x1):
			(u1, v1) = f(x1)
			(u2, v2) = f(x2)
			if v1 == v2 and u1 - u2 != k - 1:
				G.add_edge(x1, x2)
			elif u2 - u1 == k - 1 and v1 - v2 == 1:
				G.add_edge(x1, x2)
	G.add_edge(n - 1, 0)
	return G

def tree(depth, b):
	# b-ary tree of a given depth
	num_nodes = (b ** depth - 1) // (b - 1)
	num_non_leaves = (b ** (depth - 1) - 1) // (b - 1)
	G = nx.Graph([])
	for i in range(num_non_leaves):
		for j in range(b):
			G.add_edge(i, b*i+j+1)
	return G

def plot_graph(spectral_values, triangle_values):
	font = {'size'   : 20}
	matplotlib.rc('font', **font)
	fig, ax1 = plt.subplots()
	x_values = list(range(len(spectral_values)))
	ax1.plot(x_values, spectral_values, color='C0')
	ax2 = ax1.twinx()
	ax2.plot(x_values, triangle_values, color='C1')
	fig.tight_layout()
	plt.savefig('rewiring_plot.png', dpi=1200)
	plt.show()

def generate_rewiring_data(G, num_trials=1, num_iterations=250, rewiring_method='grlef'):
	# simulates rewiring of a graph for a given number of iterations, and outputs the average spectral gaps/triangle counts
	spectral_values = np.zeros(num_iterations)
	triangle_values = np.zeros(num_iterations)
	for trial in range(num_trials):
		triangle_data = None
		curvatures = None

		for i in range(num_iterations):
			if rewiring_method == "rlef":
				rewiring.rlef(G)
			elif rewiring_method == "grlef":
				triangle_data = rewiring.grlef(G, triangle_data=triangle_data)
			elif rewiring_method == "sdrf":
				G, curvatures = rewiring.sdrf(G, curvatures=curvatures, C_plus=-inf)
			spectral_gap = rewiring.spectral_gap(G)
			num_triangles = rewiring.number_of_triangles(G)
			spectral_values[i] += spectral_gap
			triangle_values[i] += num_triangles
	
	spectral_values /= num_trials
	triangle_values /= num_trials
	return spectral_values, triangle_values

if __name__ == '__main__':

	rewiring_method = "grlef"
	graph_type = "path_of_cliques"

	if graph_type == "tree":
		G = tree(depth=3, b=6)
	elif graph_type == "dumbbell":
		G = dumbbell_graph(25, 1)
	elif graph_type == "ring_of_cliques":
		G = ring_of_cliques(250, 4)
	elif graph_type == "path_of_cliques":
		G = path_of_cliques(3, 10)

	spectral_values, triangle_values = generate_rewiring_data(G, rewiring_method=rewiring_method)
	plot_graph(list(spectral_values), list(triangle_values))


