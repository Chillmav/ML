import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

class Squiggle:

    def __init__(self):

        self.slop = -2
        self.intercept = -10

    def visualize(self, data):

        x, y = data
        p = np.exp(x * self.slop + self.intercept) / (1 + np.exp(x * self.slop + self.intercept))
        plt.scatter(x, y, color="blue")
        plt.plot(x, p, color="green")
        plt.show()

    def train(self, data, epoch, lr=0.001):

        x, y = data
        p = np.exp(x * self.slop + self.intercept) / (1 + np.exp(x * self.slop + self.intercept))
        prev_log_likelihood = 0
        e = 0
        for _ in range(epoch):

            p = np.exp(x * self.slop + self.intercept) / (1 + np.exp(x * self.slop + self.intercept))
            grad_slop = np.mean((p - y) * x)
            grad_intercept = np.mean((p - y))
            log_likelihood = np.sum(
            y * np.log(p) + (1 - y) * np.log(1 - p)
            )

            self.slop -= grad_slop * lr
            self.intercept -= grad_intercept * lr
            e += 1

            if (abs(log_likelihood - prev_log_likelihood) < 1e-6):
                print("Ended after break criterium")
                break

            prev_log_likelihood = log_likelihood

        print(self.slop)
        print(self.intercept)
        print(f"epoch: {e}")
        
    def test(self, data):

        x, y = data
        p = np.exp(x * self.slop + self.intercept) / (1 + np.exp(x * self.slop + self.intercept))
        results = np.round(p)
        self.display_results(x, y, results)

        pass

    def display_results(self, x, actual: np.ndarray, predicted: np.ndarray):

        x_line = np.linspace(x.min(), x.max(), 500)
        p_line = np.exp(x_line * self.slop + self.intercept) / (1 + np.exp(x_line * self.slop + self.intercept))

        plt.scatter(x, actual, alpha=0.5)
        plt.plot(x_line, p_line, color="green", linewidth=2)

        Accuracy = metrics.accuracy_score(actual, predicted)
        Precision = metrics.precision_score(actual, predicted)
        Sensitivity_recall = metrics.recall_score(actual, predicted)
        Specificity = metrics.recall_score(actual, predicted, pos_label=0)
        F1_score = metrics.f1_score(actual, predicted)
        c_m = metrics.confusion_matrix(actual, predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=c_m)
        cm_display.plot()
        print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score}) 
        plt.show()
        
        pass


def generate_data(n, start, noise_intensity):

    x = np.arange(start, n + start) + np.random.normal(0, noise_intensity, n)
    y = np.concatenate((
        np.zeros(n // 2), # is not obese
        np.ones(n - n // 2) # is obese
    ))

    return x, y
    

data = generate_data(100, 0, 0.2)
squiggle = Squiggle()
squiggle.train(data, 1000)
squiggle.visualize(data)
squiggle.test(data=generate_data(100, -5, 1.6))
