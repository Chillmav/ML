import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class KMeans:

    def __init__(self, data, k, max_iters = 1000):

        data = data[:200]
        standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        self.data = standardized_data
        self.k = k
        self.max_iters = max_iters
        
    def fit(self):

            indices = np.random.choice(len(self.data), size=self.k, replace=False)
            distinct_points = self.data[indices]
            groups = self.data
            classes = np.zeros(self.data.shape[0])
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
                     classes[i] = np.argmin(row)
                mean_points /= points_in_clusters[:, None]
                distinct_points = mean_points
            groups = np.c_[groups, classes.reshape(-1, 1)]    
            colors = ["orange", "blue", "green", "purple", "black"]
            for point in groups:
                plt.scatter(point[0], point[1], color=colors[int(point[2])])
            plt.scatter(distinct_points[:, 0], distinct_points[:, 1], color="red", s=100)
            plt.show()
df = pd.read_csv("ML/data/SOCR-HeightWeight(1).csv")
data = df[["Height(Inches)", "Weight(Pounds)"]].values
kMeans = KMeans(data, k=5)
kMeans.fit()