"""Dieses Python File beinhaltet eine Klasse "GaussianNaiveBayes" welche einen Merkmaldatensatz und den dazugehörigen Labels eine Klassifikation nach einem 
Naiven Bayes verfahren durchführt

"""

__author__ = "Eric Hirsch und Jonas Morsch"
__version__ = "1.0.0"


import numpy as np


class GaussianNaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Berechnen des Mittelwerts, Varianz und der aPriori Wahrscheinklichkeit für jede Klasse
        for i, c in enumerate(self._classes):
            X_for_class_c = X[y == c]
            self._mean[i, :] = X_for_class_c.mean(axis=0)
            self._var[i, :] = X_for_class_c.var(axis=0)
            self._priors[i] = X_for_class_c.shape[0] / float(n_samples)

# Funktion für die Berechung der Wahrscheinlichkeiten

    def _calculate_likelihood(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        num = np.exp(- (x-mean)**2 / (2 * var))
        denom = np.sqrt(2 * np.pi * var)
        return num / denom

# classifications by calculating the posterior probability, P(H|E), of the classes
# Klassifizierung der Klassen über berechnen der posterior Wahrscheinlichkeiten
    def predict(self, X):
        y_pred = [self._classify_sample(x) for x in X]
        return np.array(y_pred)

    def _classify_sample(self, x):
        posteriors = []
        # berechnen der posterior Wahrscheinlichkeiten für jede Klasse
        for i, c in enumerate(self._classes):
            prior = np.log(self._priors[i])
            posterior = np.sum(np.log(self._calculate_likelihood(i, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
         # Klasse mit der gößten Wahrscheinlichkeit wird zurückgegeben
        return self._classes[np.argmax(posteriors)]
