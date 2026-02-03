import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

class Knn_scratch:
    
    def __init__(self, training_data, training_labels, K):
        
        self.K = K
        self.t_data = training_data
        self.t_labels = training_labels
        self.labels = None

    def predict(self, test_data):
        self.labels = np.zeros(len(test_data), dtype=int)
        for i, point in enumerate(test_data):
            distances = np.array([np.linalg.norm(t_point - point) for t_point in self.t_data])
            indices = np.argsort(distances)[:self.K]
            label = np.argmax(np.bincount(self.t_labels[indices]))
            self.labels[i] = int(label)
        print(self.labels)    




df = pd.read_csv("ML/data/SOCR-HeightWeight(1).csv")


train_data = df[["Height(Inches)", "Weight(Pounds)"]].values[:1000]
test_data = df[["Height(Inches)", "Weight(Pounds)"]].values[1000:1200] + np.random.uniform(-5, 5, (200,2))

scaler = StandardScaler()

train_data = scaler.fit_transform(train_data)
test_data = scaler.fit_transform(test_data)

kMeans = KMeans(n_clusters=6)
kMeans.fit_transform(train_data)
t_labels = kMeans.labels_

fit, axes = plt.subplots(1, 3, figsize=(12, 5))

colors = []
for i in range(np.max(t_labels) + 1):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

for (x, label) in zip(train_data, t_labels):
    axes[0].scatter(x[0], x[1], color=colors[label])

for x in test_data:
    axes[0].scatter(x[0], x[1], color="gray")

plt.tight_layout()

scratch = Knn_scratch(train_data, t_labels, K=5)

scratch.predict(test_data)
scratch_labels = scratch.labels

for (x, label) in zip(train_data, t_labels):
    axes[1].scatter(x[0], x[1], color=colors[label])

for (x, label) in zip(test_data, scratch_labels):
    axes[1].scatter(x[0], x[1], color=colors[label])


sklearnKNN = KNeighborsClassifier()

sklearnKNN.fit(train_data, t_labels)
sklearn_labels = sklearnKNN.predict(test_data)

for (x, label) in zip(train_data, t_labels):
    axes[2].scatter(x[0], x[1], color=colors[label])

for (x, label) in zip(test_data, sklearn_labels):
    axes[2].scatter(x[0], x[1], color=colors[label])

plt.show()