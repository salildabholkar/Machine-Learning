import numpy as np
from utils.CommonSetup import CommonSetup
from utils.helpers import softmax


# Binary Naive Bayes Algorithm (Gaussian)
class NaiveBayes(CommonSetup):
    n_classes = 2

    def fit(self, X, y=None):
        self._fit_model(X, y)

        self._mean = np.zeros((self.n_classes, self.n_features))
        self._var = np.zeros((self.n_classes, self.n_features))
        self._priors = np.zeros(self.n_classes)

        for c in range(self.n_classes):
            X_c = X[y == c]  # Filter features by class

            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(X.shape[0])

    def _predict(self, X=None):
        # for each row
        predictions = np.apply_along_axis(self.__predict_row, 1, X)

        # Normalize probabilities so that each row will sum up to 1.0
        return softmax(predictions)

    # likelihood for given row
    def __predict_row(self, x):
        output = []
        for y in range(self.n_classes):
            prior = np.log(self._priors[y])
            posterior = np.log(self.__pdf(y, x)).sum()
            prediction = prior + posterior

            output.append(prediction)
        return output

    # Gaussian PDF for a feature
    def __pdf(self, n_class, x):

        mean = self._mean[n_class]
        var = self._var[n_class]

        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
