import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv("ML/data/spam.csv", encoding="latin-1")
data = df[["v1", "v2"]].values

def split_dataset(data):

    size = data.shape[0]
    train_data_indices = np.random.choice(size, int(np.round(size * 4/5)), replace=False)
    test_data = np.delete(data, train_data_indices, axis=0)
    train_data = data[train_data_indices]
    return train_data, test_data

train_data, test_data = split_dataset(data)

class NaiveBaiseClassifier:

    def __init__(self, train_data, alpha=1):

        self.train_texts = train_data[:, 1]
        self.train_labels = train_data[:, 0]
        self.spam_dict = {}
        self.legit_dict = {}
        self.spam_priori = 0 
        self.legit_priori = 0 
        self.total_spam_words = 0
        self.total_legit_words = 0
        self.alpha = alpha  
        self.total_vocab = 0
    def fit(self):

        for (text, label) in zip(self.train_texts, self.train_labels):

            words = text.split(" ")

            if (label == "ham"):
                for word in words:
                    self.legit_dict[word] = self.legit_dict.get(word, 0) + 1
            else:
                for word in words:
                    self.spam_dict[word] = self.spam_dict.get(word, 0) + 1

        counter = Counter(self.train_labels)
        self.spam_priori = counter.get("spam") / (counter.get("spam") + counter.get("ham"))
        self.legit_priori = 1 - self.spam_priori
        self.total_spam_words = sum(self.spam_dict.values())
        self.total_legit_words = sum(self.legit_dict.values())
        vocab = set(self.spam_dict) | set(self.legit_dict) # union
        self.total_vocab = len(vocab)

    def predict(self, test_data):

        predictions = []
        for text in test_data:
            predictions.append(self.predict_label(text))

        return predictions
    
    def validate(self, labels, predictions):

        guessed = 0
        for (label, prediction) in zip(labels, predictions):

            if (label == prediction):
                guessed += 1

        return guessed / len(labels)

    def predict_label(self, text):

        spam_likelyhood, legit_likelyhood = np.log(self.spam_priori), np.log(self.legit_priori)
        words = text.split(" ")
        for word in words:

            spam_likelyhood += np.log((int(self.spam_dict.get(word, 0)) + self.alpha) / (self.total_vocab * self.alpha + self.total_spam_words))
            legit_likelyhood += np.log((int(self.legit_dict.get(word, 0)) + self.alpha) / (self.total_vocab * self.alpha + self.total_legit_words))

        if (spam_likelyhood > legit_likelyhood):
            return "spam"
        
        return "ham"
        

model = NaiveBaiseClassifier(train_data)
model.fit()
predictions = model.predict(test_data[:, 1])
accuracy = model.validate(test_data[:, 0], predictions)
print(accuracy)

