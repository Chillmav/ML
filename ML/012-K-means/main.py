import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class kMeans:

    def __init__(self, data, k, max_iters = 1000):

        self.data = data
        self.k = k
        self.max_iters = max_iters
        self.labels = 0
    def fit(self):

            indices = np.random.choice(len(self.data), size=self.k, replace=False)
            distinct_points = self.data[indices]
            groups = self.data
            labels = np.zeros(self.data.shape[0])
            for _ in range(self.max_iters):
                mean_points = np.zeros((self.k, 2))
                distances = []    
                for point in distinct_points:
                    if (len(distances) == 0):
                         distances = np.sqrt(np.sum((self.data - point)**2, axis=1))
                    else:
                        distances = np.c_[distances, np.sqrt(np.sum((self.data - point)**2, axis=1))]
                points_in_clusters = np.zeros(self.k)
                for (i, row) in enumerate(distances):
                     mean_points[np.argmin(row)] += self.data[i]
                     points_in_clusters[np.argmin(row)] += 1
                     labels[i] = np.argmin(row)
                mean_points /= points_in_clusters[:, None]
                distinct_points = mean_points
            self.labels = labels

    def labels_(self):
         
         return self.labels

df = pd.read_csv("ML/data/SOCR-HeightWeight(1).csv")
data = df[["Height(Inches)", "Weight(Pounds)"]].values[:500]
scaler = StandardScaler()
data = scaler.fit_transform(data)

k = 3
figure, axes = plt.subplots(1, 2, figsize=(10, 5))

kMeans = kMeans(data, k)
kMeans.fit()

KMeans = KMeans(k).fit(data)

labels_scratch = kMeans.labels_()
labels_sklearn = KMeans.labels_
print(len(labels_sklearn))
colors = []
for i in range(k):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

for point, label in zip(data, labels_scratch):
     
     axes[0].scatter(point[0], point[1], color=colors[int(label)])
axes[0].set_title("Implementation from scratch")

for point, label in zip(data, labels_sklearn):
     
     axes[1].scatter(point[0], point[1], color=colors[int(label)])
axes[1].set_title("Sklearn approach")

plt.tight_layout()
plt.show()