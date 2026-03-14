import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n = 100

x1_class1 = np.random.normal(700, 120, n)
x2_class1 = np.random.normal(7, 1.2, n)

x1_class2 = np.random.normal(400, 120, n)
x2_class2 = np.random.normal(4, 1.2, n)

X_class1 = np.column_stack((x1_class1, x2_class1))
X_class2 = np.column_stack((x1_class2, x2_class2))

x1_class1_t = np.random.normal(700, 120, n // 5)
x2_class1_t = np.random.normal(7, 1.2, n // 5)

x1_class2_t = np.random.normal(400, 120, n // 5)
x2_class2_t = np.random.normal(4, 1.2, n // 5)

X_class1_t = np.column_stack((x1_class1_t, x2_class1_t))
X_class2_t = np.column_stack((x1_class2_t, x2_class2_t))

X = np.vstack((X_class1, X_class2))
Y = np.hstack((np.ones(n), np.zeros(n)))
X_test = np.vstack((X_class1_t, X_class2_t))
Y_test = np.hstack((np.ones(n//5), np.zeros(n//5)))

# plt.scatter(X_class1[:,0], X_class1[:,1])
# plt.scatter(X_class2[:,0], X_class2[:,1])
# plt.show()

class dualSVM:

    def __init__(self, kernel="linear", epochs = 300, C=1, b=0, eps=1e-8, tol=1e-5):

        self.kernel = kernel
        self.epochs = epochs
        self.tol = tol
        self.C = C
        self.b = b
        self.eps = eps
        self.alphas = 0
        self.w = 0

        # training data
        self.X = 0
        self.Y = 0
        self.error = 0
        self.m = 0
        self.n = 0
        # standardization
        self.mean = 0
        self.std = 0

        # kernel function

        self.kernel_function = self.linear_kernel if kernel == "linear" else None

        pass

    def label_Y(self, Y):

        inherent_labels = np.unique(Y)
        if len(inherent_labels) > 2:
            raise Exception("This is binary SVM so no more than 2 classes are permitted")
        Y[Y == 0] = -1
        
        return Y
    
    def train(self, X, Y):

        self.m, self.n = np.shape(X)
        self.alphas = np.zeros(self.m)
        self.w = np.zeros(self.n)
        self.X = self.standardize(X)
        self.Y = self.label_Y(Y)
        self.error = np.array([
            self.predict(self.X[i]) - self.Y[i]
            for i in range(self.m)
        ])
        epoch = 0
        num_changed = 0
        examine_all = 1

        while (num_changed > 0 or examine_all):

            if epoch > self.epochs:
                print("Break cuz of exceeding max epochs")
                break
            num_changed = 0

            if examine_all:
                for l in range(self.m):
                    num_changed += self.examine_example(l)
            else:
                l_list = [idx for idx, value in enumerate(self.alphas) if value > 0 and value < self.C]
                for l in l_list:
                    num_changed += self.examine_example(l)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            epoch += 1

        self.plot_graph(X)
        
    def linear_kernel(self, X, x):

        return X @ x
    
    def predict(self, x):

        return self.w @ x + self.b
    
    def take_step(self, l1, l2):

        if (l1 == l2): return 0

        x1 = self.X[l1, :]
        x2 = self.X[l2, :]
        y1 = self.Y[l1]
        y2 = self.Y[l2]

        alpha1 = self.alphas[l1]
        alpha2 = self.alphas[l2]
        
        b = self.b
        s = y1 * y2

        E1 = self.calc_error(x1, y1)
        E2 = self.calc_error(x2, y2)

        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)

        if L == H:
            return 0

        k11 = self.kernel_function(x1, x1)
        k12 = self.kernel_function(x1, x2)
        k22 = self.kernel_function(x2, x2)

        eta = k11 + k22 - 2 * k12 # handling abnormal case

        if eta > 0:

            alpha2_new = alpha2 + y2*(E1 - E2)/eta
            if alpha2_new >= H:
                alpha2_new = H
            elif alpha2_new <= L:
                alpha2_new = L
        else:

            return 0 # just continue without any progress in this iteration
        
        if abs(alpha2_new - alpha2) < self.eps * (alpha2_new + alpha2 + self.eps):
            return 0
        
        alpha1_new = alpha1 + s*(alpha2 - alpha2_new) 

        b1 = b - E1 - y1*(alpha1_new - alpha1)*k11 - y2*(alpha2_new - alpha2)*k12
        b2 = b - E2 - y1*(alpha1_new-alpha1)*k12 - y2*(alpha2_new-alpha2)*k22

        if alpha1_new > 0 and alpha1_new < self.C:
            b = b1
        if alpha2_new > 0 and alpha2_new < self.C:
            b = b2
        else:
            b = (b1 + b2) / 2

        self.w = self.w + y1*(alpha1_new-alpha1)*x1 + y2*(alpha2_new - alpha2)*x2

        self.alphas[l1] = alpha1_new
        self.alphas[l2] = alpha2_new
        self.b = b

        self.error[l1] = 0
        self.error[l2] = 0

        for i in range(self.m):
            self.error[i] = self.predict(self.X[i]) - self.Y[i]
            
        return 1

    def examine_example(self, l2):

        y2 = self.Y[l2]
        x2 = self.X[l2]
        alpha2 = self.alphas[l2]
        E2 = self.error[l2]
        r2 = E2 * y2

        if ((r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0)):
            if len(self.alphas[(self.alphas > 0) & (self.alphas < self.C)]) > 1:

                if E2 > 0:
                    l1 = np.argmin(self.error)
                else:
                    l1 = np.argmax(self.error)

                if self.take_step(l1, l2):
                    return 1
                
        l1_list = [idx for idx, alpha in enumerate(self.alphas) 
                           if 0 < alpha and alpha < self.C]
        
        l1_list = np.roll(l1_list, np.random.choice(np.arange(self.m)))
        for l1 in l1_list:
            if self.take_step(l1, l2):
                return 1

        l1_list = np.roll(np.arange(self.m), np.random.choice(np.arange(self.m)))
        for l1 in l1_list:
            if self.take_step(l1, l2):
                return 1

        return 0
    

    def calc_error(self, x1, y1):

        return self.predict(x1) - y1

    def standardize(self, X, train=True):

        if train:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)

        return (X - self.mean) / self.std

    def test(self, X_test, Y_test):
        
        X_original = X_test
        X_test = self.standardize(X_test, train=False)
        Y_test = self.label_Y(Y_test)

        predictions = np.sign([
            self.predict(x) for x in X_test
        ])

        predictions = self.label_Y(predictions)

        print(np.sum(predictions == Y_test) / len(Y_test))
        self.plot_graph(X_original)
        return np.sum(predictions == Y_test) / len(Y_test)
    
    def plot_graph(self, X):

        plt.scatter(X[:,0], X[:,1])

        x1 = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)

        w1, w2 = self.w
        mu1, mu2 = self.mean
        s1, s2 = self.std

        x2 = (s2/w2) * (-(w1/s1)*(x1 - mu1) - self.b) + mu2
        
        plt.plot(x1, x2, color="green")

        # margines:

        x2_1 = x2 = (s2/w2) * (-(w1/s1)*(x1 - mu1) - self.b - 1) + mu2
        x2_2 = x2 = (s2/w2) * (-(w1/s1)*(x1 - mu1) - self.b + 1) + mu2

        plt.plot(x1, x2_1, linestyle="dashed", color="orange")
        plt.plot(x1, x2_2, linestyle="dashed", color="orange")
        plt.show()


model = dualSVM()
model.train(X, Y)
model.test(X_test, Y_test)



        