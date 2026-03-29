import numpy as np
from collections import Counter
import pandas as pd



df = pd.read_csv("Healthcare_dane_uczone.xlsx")

class Node:
    """Klasa reprezentująca pojedynczy węzeł w drzewie decyzyjnym."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature       # Indeks cechy do podziału
        self.threshold = threshold   # Wartość progowa podziału
        self.left = left             # Lewe poddrzewo
        self.right = right           # Prawe poddrzewo
        self.value = value           # Wartość klasy (tylko dla liści)

    def is_leaf_node(self):
        """Sprawdza, czy węzeł jest liściem (nie ma dzieci)."""
        return self.value is not None


class DecisionTree:
    """Główna klasa klasyfikatora Drzewa Decyzyjnego."""
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split # Min. liczba próbek do podziału
        self.max_depth = max_depth                 # Maksymalna głębokość drzewa
        self.n_features = n_features               # Liczba cech branych pod uwagę przy podziale
        self.root = None                           # Korzeń drzewa

    def fit(self, X, y):
        """Trenuje model na danych treningowych."""
        # Upewniamy się, że nie wybieramy więcej cech niż istnieje w zbiorze
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """Rekurencyjnie buduje drzewo decyzyjne."""
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Warunki stopu (liść): max głębokość, jednorodność klas, lub za mało próbek
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Wybór losowych cech do sprawdzenia (zapobiega przeuczeniu, ważne m.in. w Random Forest)
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Znalezienie najlepszego podziału
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Budowa dzieci (lewe i prawe poddrzewo)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        """Znajduje optymalną cechę i próg podziału maksymalizujący zysk informacyjny."""
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        """Oblicza zysk informacyjny (Information Gain) dla danego podziału."""
        # Entropia rodzica
        parent_entropy = self._entropy(y)

        # Generowanie indeksów po podziale
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Obliczanie średniej ważonej entropii dzieci
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Zysk informacyjny
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        """Dzieli zbiór na dwie części na podstawie progu."""
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """Oblicza entropię wektora klas."""
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """Zwraca najczęściej występującą klasę (głosowanie większościowe w liściu)."""
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """Przewiduje etykiety dla nowych danych."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Przechodzi przez drzewo dla pojedynczej próbki w celu dokonania predykcji."""
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)