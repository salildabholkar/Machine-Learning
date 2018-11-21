import numpy as np
import math


def euclidean_distance(a, b):
    if isinstance(a, list) and isinstance(b, list):
        a = np.array(a)
        b = np.array(b)

    return math.sqrt(sum((a - b) ** 2))


def l2_distance(X):
    sum_X = np.sum(X * X, axis=1)
    return (-2 * np.dot(X, X.T) + sum_X).T + sum_X


def softmax(z):
    # Avoid numerical overflow by removing max
    e = np.exp(z - np.amax(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def classification_error(actual, predicted):
    return (actual != predicted).sum() / float(actual.shape[0])


def accuracy(actual, predicted):
    return 1.0 - classification_error(actual, predicted)
