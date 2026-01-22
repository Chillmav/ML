"""
Ridge Regression:

-The main idea behind Ridge Regression is to find a #NEW LINE# that doesn`t fit the Training Data as well as for example linear regression, but we prevent our model from overfitting to training data. 

-In other words, we introduce a small amount of BIAS into how the NEW LINE is fit to the data.

-In return for that small amount of Bias we get a significant drop in Variance.

-By starting with a slightly worse fit, RIDGE REGRESSION can provide better long term predictions.

In rigde regression we minimizes:

y = ax + b

The sum of the squared residuals +  λ * slope**2

λ -> (0, +inf)

higher the lambda is, lower differences in y caused by bigger.

"""


import numpy as np
import matplotlib.pyplot as plt

class Line:

    def __init__(self):

        self.slope = 0 
        self.intercept = 10 

    def visualize_fit(self, data):

        x, y = data


    def least_squares_ridge(self, data, lambdas=np.logspace(-3, 3, 20), cross_validation=10):

        x, y = data
        data_length = len(x)
        seed = np.random.randint(0, 10000)
        np.random.seed(seed)  
        np.random.shuffle(x)  
        np.random.seed(seed)  
        np.random.shuffle(y)  


        fold = data_length // cross_validation
        prev_avg_error = 1e6
        best_λ = lambdas[0]

        for λ in lambdas:
            avg_error = 0

            for i in range(cross_validation):

                start = i * fold
                end = start + fold

                x_test = x[start:end]
                y_test = y[start:end]

                x_train = np.concatenate((x[:start], x[end:]))
                y_train = np.concatenate((y[:start], y[end:]))


                X = np.c_[x_train, np.ones(len(x_train))]
                theta = np.linalg.inv(
                    X.T @ X + λ * np.array([[1,0], [0, 0]])
                ) @ X.T @ y_train

                slope = theta[0]
                intercept = theta[1]
                
                avg_error += np.sum((y_test - (x_test * slope + intercept))**2)
                visualize(slope, intercept, data, "gray")     

            avg_error /= cross_validation
            
            if (avg_error < prev_avg_error):
                best_λ = λ
            
            prev_avg_error = avg_error

        X_full = np.c_[x, np.ones(len(x))]
        theta = np.linalg.inv(
            X_full.T @ X_full + best_λ * np.array([[1,0],[0,0]])
        ) @ X_full.T @ y

        self.slope, self.intercept = theta

        plt.scatter(x, y, color="blue")
        visualize(self.slope, self.intercept, data, "green")
        print(f"Best λ is equal to {best_λ}")
        plt.show()

def generate_data(n):
    x = np.arange(n) + np.random.normal(0, 1, n)
    y = 2 * np.arange(n) + np.random.normal(0, 1, n)
    return x, y

def visualize(slope, intercept, data, color):

    x, _ = data
    plt.plot(x, x * slope + intercept, color=color)


line = Line()

line.least_squares_ridge(generate_data(20))