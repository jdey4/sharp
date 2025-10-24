import numpy as np
import networkx as nx
from scipy.spatial import distance

class GrowingNeuralGas:
    def __init__(self, data):
        self.data = data
        self.graph = nx.Graph()
        self.errors = {}

    def fit_network(self, e_b=0.05, e_n=0.006, a_max=15, l=100, a=0.5, d=0.995, passes=10, plot_evolution=False):
        # initialize with two random nodes
        idx = np.random.choice(range(len(self.data)), 2, replace=False)
        for i in idx:
            self.graph.add_node(i, vector=self.data[i], error=0.0)
        self.graph.add_edge(idx[0], idx[1], age=0)

        t = 0
        for epoch in range(passes):
            for i, x in enumerate(self.data):
                # Find 2 closest nodes
                dists = {n: distance.euclidean(x, self.graph.nodes[n]['vector']) for n in self.graph.nodes}
                s_1, s_2 = sorted(dists, key=dists.get)[:2]

                # Increase age of edges from s_1
                for neighbor in list(self.graph.neighbors(s_1)):
                    self.graph.edges[s_1, neighbor]['age'] += 1

                # Update error
                self.graph.nodes[s_1]['error'] += distance.euclidean(x, self.graph.nodes[s_1]['vector']) ** 2

                # Move s_1 and its neighbors
                self.graph.nodes[s_1]['vector'] += e_b * (x - self.graph.nodes[s_1]['vector'])
                for neighbor in self.graph.neighbors(s_1):
                    self.graph.nodes[neighbor]['vector'] += e_n * (x - self.graph.nodes[neighbor]['vector'])

                # Add or reset edge between s_1 and s_2
                if self.graph.has_edge(s_1, s_2):
                    self.graph.edges[s_1, s_2]['age'] = 0
                else:
                    self.graph.add_edge(s_1, s_2, age=0)

                # Remove old edges
                for u, v, attrs in list(self.graph.edges(data=True)):
                    if attrs['age'] > a_max:
                        self.graph.remove_edge(u, v)
                        if self.graph.degree[u] == 0:
                            self.graph.remove_node(u)
                        if self.graph.degree[v] == 0:
                            self.graph.remove_node(v)

                # Insert new node every l steps
                if t % l == 0 and len(self.graph.nodes) < len(self.data):
                    # node with highest error
                    q = max(self.graph.nodes, key=lambda n: self.graph.nodes[n]['error'])
                    # neighbor of q with highest error
                    neighbors = list(self.graph.neighbors(q))
                    if not neighbors:
                        continue
                    f = max(neighbors, key=lambda n: self.graph.nodes[n]['error'])
                    # insert new node r between q and f
                    v_r = 0.5 * (self.graph.nodes[q]['vector'] + self.graph.nodes[f]['vector'])
                    r_id = max(self.graph.nodes) + 1
                    self.graph.add_node(r_id, vector=v_r, error=0.0)
                    self.graph.add_edge(r_id, q, age=0)
                    self.graph.add_edge(r_id, f, age=0)
                    self.graph.remove_edge(q, f)
                    self.graph.nodes[q]['error'] *= a
                    self.graph.nodes[f]['error'] *= a

                # Decay all errors
                for n in self.graph.nodes:
                    self.graph.nodes[n]['error'] *= d

                t += 1

    def cluster_data(self):
        # Assign each point to closest GNG node
        centroids = [self.graph.nodes[n]['vector'] for n in self.graph.nodes]
        assignments = []
        for x in self.data:
            distances = [distance.euclidean(x, c) for c in centroids]
            assignments.append(np.argmin(distances))
        return assignments
