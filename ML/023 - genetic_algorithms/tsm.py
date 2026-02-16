import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class TSMGenetic:

    def __init__(self, pop_size=60, n=10, epochs=500, p_m=0.10):

        self.pop_size = pop_size
        self.n = n
        self.epochs = epochs
        self.cities = self.init_cities()
        self.population = []
        self.p_m = p_m

    def init_cities(self):

        return np.random.uniform(-10, 10, (self.n, 2))

    def visualize(self):

        offset = 0.5
        for i, city in enumerate(self.cities):

            plt.scatter(city[0], city[1])
            plt.text(city[0] - offset, city[1], f'{i}')

    def train(self):

        self.visualize()
        self.init_pop()

        for e in range(self.epochs):

            distances = self.compute_total_distances()
            if (e % 50 == 0):
                print(f"Mean distance in epoch {e}: {np.mean(distances)}")
            surviviors = self.selection(distances)
            offspring = self.crossover(surviviors)
            new_population = np.concatenate((surviviors, offspring))
            self.mutate(new_population)
            self.population = new_population

        distances = self.compute_total_distances()
        print(self.population)
        print(self.population[np.argmin(distances)])

    def selection(self, distances): 

        half = int(self.pop_size / 2)
        survivors = np.zeros((half, self.n), dtype=int)

        for i in range(half):

            distance_1 = distances[i]
            distance_2 = distances[i + half]

            if (distance_1 <= distance_2):

                survivors[i] = self.population[i]
            else:
                survivors[i] = self.population[i + half]
        
        return survivors

    def crossover(self, survivors):

        half = int(len(survivors) / 2)
        offspring = np.zeros((len(survivors), self.n), dtype=int)

        for i in range(half):

            path_1 = survivors[i]
            path_2 = survivors[i + half]
            offspring_1 = self.cross(path_1, path_2)
            offspring_2 = self.cross(path_2, path_1)

            offspring[i] = offspring_1
            offspring[i+half] = offspring_2

        return offspring
    
    def cross(self, p1, p2):

        n = 3
        start = np.random.randint(1, self.n - n + 1)
        offspring = np.full(self.n, fill_value=-1,dtype=int)
        offspring[start:start+n] = p2[start:start+n]
        k = 0
        for j in range(len(offspring)):
            
            if offspring[j] == -1:
                while p1[k] in offspring:
                    k += 1
                offspring[j] = p1[k]
                k += 1

        return offspring

    def mutate(self, new_population):

        for i in range(len(new_population)):

            if (np.random.rand() < self.p_m):
                indices = np.random.choice(np.arange(1, self.n), 2, replace=False)

                new_population[i][indices[0]], new_population[i][indices[1]] = new_population[i][indices[1]], new_population[i][indices[0]]

    def init_pop(self):

        self.population = np.zeros((self.pop_size, self.n), dtype=int)

        for i in range(self.pop_size):
            path = np.arange(1, self.n, dtype=int)
            np.random.shuffle(path)
            self.population[i, 0] = 0
            self.population[i, 1:] = path
        

    def compute_total_distances(self):

        distances = np.zeros(self.pop_size)

        for i, path in enumerate(self.population):
            
            distances[i] = self.compute_dist(path)

        return distances

    def compute_dist(self, path):

        coords = self.cities[path]              
        rolled = np.roll(coords, -1, axis=0)    

        total_dist = np.sum(
            np.linalg.norm(coords - rolled, axis=1)
        )

        return total_dist

model = TSMGenetic()

model.train()
plt.show()