import numpy as np


class GaussianNaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # calculating the mean, variance and prior P(H) for each class
        for i, c in enumerate(self._classes):
            X_for_class_c = X[y == c]
            self._mean[i, :] = X_for_class_c.mean(axis=0)
            self._var[i, :] = X_for_class_c.var(axis=0)
            self._priors[i] = X_for_class_c.shape[0] / float(n_samples)
# function for calculating the likelihood, P(E|H), of data X given the mean and variance

    def _calculate_likelihood(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        num = np.exp(- (x-mean)**2 / (2 * var))
        denom = np.sqrt(2 * np.pi * var)
        return num / denom

# classifications by calculating the posterior probability, P(H|E), of the classes
    def predict(self, X):
        y_pred = [self._classify_sample(x) for x in X]
        return np.array(y_pred)

    def _classify_sample(self, x):
        posteriors = []
        # calculating posterior probability for each class
        for i, c in enumerate(self._classes):
            prior = np.log(self._priors[i])
            posterior = np.sum(np.log(self._calculate_likelihood(i, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
         # return the class with highest posterior probability
        return self._classes[np.argmax(posteriors)]
