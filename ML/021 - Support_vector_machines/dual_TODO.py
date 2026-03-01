import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class dualSVM:

    def __init__(self, kernel="linear", epochs = 300):

        self.kernel = kernel
        self.epochs = epochs
        pass


    def train(self, X, Y):

        if self.kernel == "linear": # then k(x1,x2) = np.dot(x1, x2)

            pass




model = dualSVM()



        