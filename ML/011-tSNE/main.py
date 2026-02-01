import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def prepare_data():

    df = pd.read_csv("ML/data/insurance.csv")
    data = df[["age", "bmi", "children"]].values
    sex = df["sex"].map({
        "female": 1,
        "male": 0
    }).values
    isSmoking = df["smoker"].map({
        "yes": 1,
        "no": 0
    }).values
    region = df["region"].map({
        "southwest": 0,
        "northwest": 1,
        "southeast": 2,
        "northeast": 3
    }).values

    charges = df["charges"].values
    x = np.c_[data, sex, isSmoking, region, charges]
    
    tSNE = TSNE(2)
    X = tSNE.fit_transform(x)
    plt.scatter(X[:, 0], X[:, 1], s=10)

    plt.show()
    
prepare_data()