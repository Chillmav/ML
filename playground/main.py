import numpy as np
import pandas as pd


C = np.array([[0.8, 0.3, 0.2], [0.1, 0.2, 0.6], [0.1, 0.5, 0.2]])

eigenval, eigenvec = np.linalg.eig(C)
eigenval, eigenvec = eigenval[0], eigenvec[:, 0]
print(eigenvec / np.sum(eigenvec))
print(eigenval)


