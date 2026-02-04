import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

scaler = StandardScaler()
df = pd.read_csv("ML/data/obesity_data.csv")
# np.random.seed(42)

def split_dataset(data: np.ndarray):

    size = data.shape[0]
    train_data_indices = np.random.choice(size, int(np.round(size * 4/5)), replace=False)
    test_data = np.delete(data, train_data_indices, axis=0)
    train_data = data[train_data_indices]
    return train_data, test_data

def preprocessing(df: pd.DataFrame):

    labels = df["ObesityCategory"].map({
        "Normal weight": 0,
        "Obese": 1,
        "Overweight": 2,
        "Underweight": 3
    }).values

    gender = df["Gender"].map({
        "Female": 1,
        "Male": 0
    }).values

    data = df[["Age", "Height", "Weight", "BMI", "PhysicalActivityLevel"]]
    data = np.c_[data, gender]
    data = np.c_[data, labels]

    train_data, test_data = split_dataset(data)

    return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]

train_data, train_labels, test_data, test_labels = preprocessing(df)


class GaussianNaiveBayesClassifier:

    def __init__(self, train_data, train_labels: np.ndarray, var=1e-9):

        self.train_data = train_data
        self.train_labels = train_labels
        self.n_labels, self.priors = np.unique(train_labels, return_counts=True)
        self.priors = self.priors / train_labels.shape[0]
        self.stats = []
        self.var = var
        

    def predict(self, test_data: np.ndarray):

        predictions = np.zeros(test_data.shape[0])

        for i, x in enumerate(test_data):
            prediction = self.calc_gauss(x)
            predictions[i] = int(prediction)
        return predictions
    
    def fit(self):

        for label in self.n_labels:

            indices = np.where(self.train_labels == label)
            x = self.train_data[indices]
            self.stats.append([np.mean(x, axis=0), np.var(x, axis=0)])

    def calc_gauss(self, x):

        likelyhoods = np.zeros(len(self.n_labels))

        for l in range(len(self.n_labels)):
            likelyhood = 0
            gauss = np.exp((-1/2) * ((x - self.stats[l][0])**2 / (self.stats[l][1]))) / (np.sqrt((self.stats[l][1] + self.var)* 2*np.pi)) 
            likelyhood += np.sum(np.log(gauss))
            likelyhood += np.log(self.priors[l])
            likelyhoods[l] = likelyhood

        return np.argmax(likelyhoods)

    def validate(self, predictions, labels):

        guessed = 0
        size = len(predictions)

        for (prediction, label) in zip(predictions, labels):

            if (prediction == label):
                guessed += 1
        
        return guessed / size

model = GaussianNaiveBayesClassifier(train_data, train_labels)
model.fit()
predictions = model.predict(test_data)
accuracy = model.validate(predictions, test_labels)
print(f"My model: {accuracy}")


model_sklearn = GaussianNB(var_smoothing=1e-10)

predictions = model_sklearn.fit(train_data, train_labels).predict(test_data)

accuracy = accuracy = model.validate(predictions, test_labels)
print(f"Sklearn model: {accuracy}")