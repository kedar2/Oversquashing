import rewiring
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from math import inf

font = {'size'   : 15}

matplotlib.rc('font', **font)

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

n = 250
d = 4

G = dumbbell_graph(25, 1)
x_values = []
spectral_values = []
triangle_values = []
curvatures=None

fig, ax1 = plt.subplots()

for i in range(500):
	x_values.append(i)
	#curvature = rewiring.average_curvature(G, curvatures=curvatures)
	_, curvatures = rewiring.sdrf(G, curvatures=curvatures, C_plus=0)
	spectral_gap = rewiring.spectral_gap(G)
	num_triangles = rewiring.number_of_triangles(G)
	spectral_values.append(spectral_gap)
	triangle_values.append(num_triangles)
	#curvature_values.append(curvature)
	print(i, spectral_gap, num_triangles, len(G.edges))
	#print(i, spectral_gap, curvature, rewiring.average_curvature(G, curvatures=None))
	#(G, curvatures) = rewiring.sdrf(G, curvatures=curvatures, temperature=1000, C_plus=inf)




ax1.set_xlabel('Number of iterations')
ax1.set_ylabel('Spectral gap', color='tab:red')
ax1.plot(x_values, spectral_values, color='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Triangle count', color='tab:blue')
ax2.plot(x_values, triangle_values, color='tab:blue')
plt.title("SDRF on Dumbbell Graph")
plt.legend(loc='best')

fig.tight_layout()
plt.savefig('filename.png', dpi=1200)
plt.show()
