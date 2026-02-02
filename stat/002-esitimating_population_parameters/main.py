import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("stat/data/SOCR-HeightWeight.csv")
data = df["Height(Inches)"].values


def sample_data(data, data_samples):

    indices = np.random.choice(len(data), data_samples, replace=False) # returns choosen rows
    sample = data[indices]
    data = np.delete(data, indices)
    print(len(data))
    return data, sample

def estimate_population_parameters(samples):

    sample_mean = np.mean(samples)
    sample_var = np.sum((samples - sample_mean)**2) / (len(samples) - 1) 
    x_samples = np.linspace(np.min(samples), np.max(samples), 300)

    return sample_mean, sample_var, x_samples



def plot_results(data, estimations=3):

    figure, axes = plt.subplots(1, estimations + 1, figsize=(15, 10))
    x_data = np.linspace(np.min(data), np.max(data), 300)

    population_mean = np.mean(data)
    population_var = np.var(data)

    axes[0].hist(data, density=True)
    axes[0].plot(x_data, gauss(x_data, population_var, population_mean))
    observations = np.array([])
    for e in range(estimations):
        data, sample = sample_data(data, 836)
        observations = np.append(observations, sample)
        sample_mean, sample_var, _ = estimate_population_parameters(observations)
        axes[e+1].hist(observations, bins=10, density=True)
        axes[e+1].plot(x_data, gauss(x_data, sample_var, sample_mean))
    plt.tight_layout()
    plt.show()


def gauss(data, var, mean):

    return np.exp(-(data - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    


plot_results(data)