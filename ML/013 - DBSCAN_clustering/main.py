import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint
from sklearn.cluster import DBSCAN
rng = np.random.default_rng(42)

# Three dense clusters
cluster_1 = rng.normal(loc=[0, 0], scale=0.2, size=(150, 2))
cluster_2 = rng.normal(loc=[3, 3], scale=0.25, size=(180, 2))
cluster_3 = rng.normal(loc=[0, 4], scale=0.2, size=(120, 2))

# Uniform noise
noise = rng.uniform(low=-1, high=5, size=(50, 2))

X = np.vstack([cluster_1, cluster_2, cluster_3, noise]) # stacks arrays vertically

fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].scatter(X[:, 0], X[:, 1])

class dBSCAN:

    def __init__(self, radius, close_points):

        self.r = radius
        self.n = close_points
        self.labels = 0

    def fit(self, data):

        n_close_points, is_core_point = self.find_core_points(data)
        index = 0

        clusters = np.zeros(len(data), dtype=int)
        for i in range(len(is_core_point)):
            if (is_core_point[i] == 1):
                index = i
                break
        cluster_number = 1      

        while (np.count_nonzero(is_core_point) > 0):
            self.assing_core(data, is_core_point, index, cluster_number, clusters)
            cluster_number += 1
            for i in range(len(is_core_point)):
                if (is_core_point[i] == 1):
                    index = i
                    break
        print(clusters)


        #### assing non-core
        self.assign_non_core(data, clusters)
        print(clusters)
        ####

        self.labels = clusters

    def find_core_points(self, data):
        
        n_close_points = np.zeros(len(data), dtype=int)
        is_core_point = np.zeros(len(data), dtype=int)

        for i in range(len(data)):
            for j in range(len(data)):
                if (i == j):
                    continue
                if (np.linalg.norm(data[i] - data[j]) <= self.r):
                    n_close_points[i] += 1
                    if (n_close_points[i] == self.n):
                        is_core_point[i] = 1

        return n_close_points, is_core_point
    
    def assing_core(self, data, is_core_point, index, cluster_number, clusters):
        clusters[index] = cluster_number
        for i in range(len(is_core_point)):
            if (is_core_point[i] == 1):
                if (np.linalg.norm(data[index] - data[i]) <= self.r):
                    clusters[i] = cluster_number
                    is_core_point[i] = 0
                    self.assing_core(data, is_core_point, i, cluster_number, clusters)

    def assign_non_core(self, data, clusters):
        for i in range(len(clusters)):
            if (clusters[i] == 0):
                for j in range(len(data)):
                    if (i == j):
                        continue
                    if (np.linalg.norm(data[i] - data[j]) <= self.r and clusters[j] != 0):
                        clusters[i] = clusters[j]
                        break
        
dbscan = dBSCAN(radius=0.2, close_points=3)
Dbscan = DBSCAN(0.2)
Dbscan.fit(X)
sklearn_labels = Dbscan.labels_

dbscan.fit(X)

labels = dbscan.labels
colors = []
k = np.max(labels) + 1
for i in range(k):
    colors.append('#%06X' % randint(0, 0xFFFFFF))


for (x, label) in zip(X, labels):

    axes[1].scatter(x[0], x[1], color=colors[label])
axes[1].set_title("DBSCAN from scratch")


for (x, label) in zip(X, sklearn_labels):

    axes[2].scatter(x[0], x[1], color=colors[label])
axes[2].set_title("DBSCAN from sklearn")

plt.tight_layout()
plt.show()

