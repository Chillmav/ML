import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def normal():

    def calc_normal_distribution(x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))
    
    df = pd.read_csv("stat/data/SOCR-HeightWeight.csv")

    data = df[["Height(Inches)", "Weight(Pounds)"]].values

    figure, axes = plt.subplots(1, 2, figsize=(10,5))

    height, weight = data.T

    height = height * 2.54
    weight = weight * 0.45359237
    x_h_mean, x_h_var = np.mean(height), np.var(height)
    x_w_mean, x_w_var = np.mean(weight), np.var(weight)

    x_h, x_w = np.linspace(np.min(height), np.max(height), 300), np.linspace(np.min(weight), np.max(weight), 300)

    axes[0].hist(height, bins=20, density=True)
    axes[0].plot(x_h, calc_normal_distribution(x_h, x_h_mean, x_h_var))
    axes[1].hist(weight, bins=20, density=True)
    axes[1].plot(x_w, calc_normal_distribution(x_w, x_w_mean, x_w_var))

def exponential(): 

    x = np.random.exponential(scale=2.0, size=1000)
    x_n = np.linspace(np.min(x), np.max(x), 300)

    def estimate_rate(x):
        
        n = len(x)
        sum = np.sum(x)
        
        return n / sum
    
    def pdf(x, rate):

        return rate * np.exp(-rate * x)

    figure, axes = plt.subplots(1, 2, figsize=(10,5))
    rate = estimate_rate(x)
    print(rate)
    axes[0].hist(x, bins=10, density=True)
    axes[0].plot(x_n, pdf(x_n, rate))

    
exponential()
plt.show()
