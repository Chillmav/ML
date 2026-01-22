import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plane:

    def __init__(self):
        
        self.slope_x = 1
        self.slope_y = 0
        self.c = 0


    def fit(self, data):

        x, y, z = data
        x = np.c_[x, y, np.ones(len(x))]
        self.slope_x, self.slope_y, self.c = np.linalg.inv(x.T @ x) @ x.T @ z
    
    def visualize(self, data):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_actual, y_actual, z_actual = data
        ax.scatter(x_actual, y_actual, z_actual, color='blue', label='Data')

        x_range = np.linspace(x_actual.min(), x_actual.max(), 20)
        y_range = np.linspace(y_actual.min(), y_actual.max(), 20)
        X, Y = np.meshgrid(x_range, y_range)

        Z = self.slope_x * X + self.slope_y * Y + self.c

        ax.plot_surface(X, Y, Z, alpha=0.5, color='red')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.show()

def generate_data(n):

    x = np.arange(n) + np.random.normal(0, 1)
    y = (np.arange(-2, n - 2) + np.random.normal(0, 1)) * 1.2
    z = x * y + np.random.normal(0, 1)
    return x, y, z

data=generate_data(20)
plane = Plane()
plane.fit(data)
plane.visualize(data)