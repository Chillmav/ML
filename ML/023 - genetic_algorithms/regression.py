import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# X = [samples, features]

df = pd.read_csv("ML/data/SOCR-HeightWeight(1).csv")

X = df["Height(Inches)"]
Y = df["Weight(Pounds)"]

data = np.c_[X, Y]

def split_dataset(data: np.ndarray):

    size = data.shape[0]
    train_data_indices = np.random.choice(size, int(np.round(size * 4/5)), replace=False)
    test_data = np.delete(data, train_data_indices, axis=0)
    train_data = data[train_data_indices]
    return train_data, test_data

train_data, test_data = split_dataset(data)
X, Y, X_test, Y_test = train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]

class GeneticRegression:

    def __init__(self, pop_size=100, degree=2, epochs=1000, cross_prob=0.6, mut_prob=0.15):

        self.degree = degree
        self.pop_size = pop_size
        self.population = np.zeros((pop_size, self.degree + 1)) # weights + bias
        self.epochs = epochs
        self.p_c = cross_prob
        self.p_r = 1 - self.p_c
        self.p_m = mut_prob

        self.init_pop()

    def fitness(self, X, Y):

        powers = np.arange(self.degree + 1)
        errors= np.zeros(self.population.shape[0])
        Phi = X**(powers)

        for i in range(self.population.shape[0]):
            y_hat = Phi @ self.population[i]
            SSR = np.sum((y_hat  - Y)**2)
            errors[i] = SSR

        fitness = 1 / (1 + errors)
        probs = fitness / np.sum(fitness)

        return errors, fitness, probs
    
    def fit(self, X, Y):

        new_population = np.zeros_like(self.population)
        best = 0
        for e in range(self.epochs):

            errors, fitness, probs = self.fitness(X, Y)
            cums = np.cumsum(probs)
            self.selection(cums, new_population, errors)
            if (e % 100 == 0):

                print(f"Mean error in epoch {e}: {np.mean(errors)}")
            self.population = new_population
            


    def selection(self, cums, new_population, errors):
        
        best_idx = np.argmin(errors)
        new_population[0] = self.population[best_idx]
        i = 1
        while i < self.pop_size:

            idxs = np.zeros(2)
            idxs[0], idxs[1] = np.searchsorted(cums, np.random.rand()), np.searchsorted(cums, np.random.rand())
            r = np.random.rand()
            new_ind = 0
            if (r < self.p_c):  

                #child
                new_ind = self.crossover(self.population[int(idxs[0])], self.population[int(idxs[1])])

            else:

                #reproduction
                new_ind = self.population[int(np.random.choice(idxs))]

            if (np.random.rand() < self.p_m):

                new_ind = self.mutate(new_ind)

            new_population[i] = new_ind
            i += 1
        
    def crossover(self, p1, p2):

        alpha = np.random.rand()
        return alpha * p1 + (1 - alpha) * p2

    def mutate(self, ind):

        w = np.random.randint(0, self.degree + 1)

        ind[w] += np.random.normal(0, 0.05)

        return ind
    
    def init_pop(self):

        for s in range(self.population.shape[0]):

            self.population[s] = np.random.uniform(-5, 5, self.degree + 1)

    def calc_best_Y(self, X):

        powers = np.arange(self.degree + 1)
        Phi = X**(powers)
        return Phi @ self.population[0]

model = GeneticRegression(degree=1)
model.fit(X, Y)

X_test = np.linspace(np.min(X_test), np.max(X_test), len(X_test)).reshape(-1, 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].scatter(X, Y, color="blue")
axes[0].set_title("Real Data")
axes[1].scatter(X_test, model.calc_best_Y(X_test), color="green")
axes[1].set_title("Predicted Data")
plt.show()
