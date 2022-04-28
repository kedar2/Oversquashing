import torch
from random import random

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


def rlef_primitive(edge_index, edge_weights=None):
	# algorithm 1 from paper
	if edge_weights == None:
		edge_weights = edge_index ** 0
	pass
