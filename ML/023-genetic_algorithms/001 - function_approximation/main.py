import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


X = np.linspace(0, 10, 100)
Y = X**2 + 5*X + 2

class GeneticAlgorithm: # works for y_i >= 0 for each y_i belongs to Y

    def __init__(self, epochs=1000):

        self.cells = 0
        self.length = 0
        self.population = []
        self.epochs = epochs

    def train(self, X, Y):

        self.cells = len(X)
        self.length = self.calc_cell_length(np.max(Y))
        self.population = self.init_cells()
        self.elitism_param = int(np.round(len(self.population) / 50)) 

        for _ in range(self.epochs):

            new_population = np.zeros((self.cells, self.length))
            errors = self.cost_function(Y)
            sorted_indicies = np.argsort(errors)
            fitness = 1 / (errors + 1e-8)
            probabilities = fitness / np.sum(fitness)
            probabilities = probabilities[sorted_indicies]

            # elitism:
            sorted_indicies, probabilities = self.elitism(new_population, sorted_indicies, probabilities)

            # replication:

            sorted_indicies, probabilities = self.rep_cross_mutate(new_population, sorted_indicies, probabilities)

            self.population = new_population

        self.result(X, Y)

    def calc_probabilities(self, errors):

        return errors / np.max(errors)
    
    def init_cells(self):
        
        cells = np.zeros((self.cells, self.length))

        for i in range(cells.shape[0]):
            cells[i] = self.gen_cell() 

        return cells
    
    ## Evolutionary Mechanisms:

    def elitism(self, new_population, indices, probabilities):

        elite_indices = indices[:self.elitism_param]
        new_population[elite_indices] = self.population[elite_indices]

        return indices[self.elitism_param:], probabilities[self.elitism_param:]
    
    def rep_cross_mutate(self, new_population, indices: np.ndarray, probabilities):

        mask = np.ones(len(indices), dtype=bool)
        cross_cells = [] # 2x3 [cell, idx, i]

        for i, (idx, prob) in enumerate(zip(indices, probabilities)):

            num = np.random.uniform(0, 1)

            if (prob > num):

                new_population[idx] = self.population[idx]
                mask[i] = False

            elif (3 * prob > num):

                cross_cells.append([self.population[idx].copy(), idx, i])

                if (len(cross_cells) == 2):

                    cross_cells = self.crossover(cross_cells)
                    new_population[cross_cells[0][1]] = cross_cells[0][0]
                    new_population[cross_cells[1][1]] = cross_cells[1][0]
                    mask[cross_cells[0][2]] = False
                    mask[cross_cells[1][2]] = False
                    cross_cells.clear()
            else:

                new_population[idx] = self.mutate(self.population[idx])
                mask[i] = False

        if (len(cross_cells) == 1):

            idx = cross_cells[0][1]
            i = cross_cells[0][2]
            new_population[idx] = self.mutate(cross_cells[0][0])
            mask[i] = False
            cross_cells.clear()

            # mutate this one gene

        return indices[mask], probabilities[mask]
    

    def crossover(self, cross_cells: list):

        # Two points crossover
        points = np.sort(np.random.choice(self.length, 2, replace=False))

        for i in range(points[0], points[1]):

            cross_cells[0][0][i], cross_cells[1][0][i] = cross_cells[1][0][i], cross_cells[0][0][i]
        
        return cross_cells
    

    def mutate(self, cell):

        new_cell = cell.copy()
        point = np.random.choice(self.length)
        gene = new_cell[point] 

        new_cell[point] = 1 if gene == 0 else 0

        return new_cell

    ##

    def calc_cell_length(self, max_y):

        length = 1
        max_sum = 0
        while max_sum < max_y:
            max_sum += 2**(length-1)
            length += 1
        return length - 1

    def cost_function(self, Y):
        
        errors = np.abs(Y - self.calc_y_cells())
        return errors

    def gen_cell(self):

        cell = np.zeros(self.length)

        for i in range(self.length):
            val = 1 if np.random.uniform(-1, 1) >= 0 else 0
            cell[i] = val
        
        return cell
    
    def calc_y_cells(self):

        y_cells = np.zeros(self.cells)

        for i, cell in enumerate(self.population):
            
            y_cells[i] = self.eval_cell(cell)

        return y_cells
    
    def eval_cell(self, cell: np.ndarray):

        cell = cell[:-1]
        y = 0
        powers = 2 ** np.arange(len(cell))
        return np.sum(cell * powers)

    def result(self, x, y):

        fig, axis = plt.subplots(1, 2, figsize=(10, 5))
        axis[0].scatter(x, y, color="blue")
        axis[0].set_title("Real Function")
        
        axis[1].scatter(x, self.calc_y_cells(), color="green")
        axis[1].set_title("Approximation")

        plt.show()

model = GeneticAlgorithm()

model.train(X, Y)


