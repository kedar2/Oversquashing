import rewiring
import networkx as nx
import matplotlib.pyplot as plt

combos = [(10, 3), (50, 3), (100, 3)]
combos_2 =  [(50, 10), (100, 10), (200, 10)]
x_data = []
y_data = []
l = []
def dumbbell_graph(n, dumbbell_length):
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
for k in range(len(combos_2)):
	print(k)
	input()
	(n, dumbbell_length) = combos_2[k]
	G = dumbbell_graph(n, dumbbell_length)
	x_values = []
	y_values = []
	for i in range(20000):
		if i % 500 == 0:
			x_values.append(i)
			spectral_gap = rewiring.average_curvature(G)
			print(spectral_gap)
			y_values.append(spectral_gap)
			
		rewiring.rlef(G)
	plt.plot(x_values, y_values, label= "Clique size " + str(n))
plt.title("Handle size 10")
plt.legend(loc='best')
plt.xlabel("Number of iterations")
plt.ylabel("Total curvature")
plt.show()