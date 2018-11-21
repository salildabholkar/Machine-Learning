from collections import Counter

import numpy as np
from utils.helpers import euclidean_distance

from utils.CommonSetup import CommonSetup


class KNN(CommonSetup):
    def __init__(self, k=5):
        self.k = k  # l[:None] returns the whole list

    def get_most_common(self, neighbors_targets):
        return Counter(neighbors_targets).most_common(1)[0][0]

    def _predict(self, X=None):
        predictions = [self.__predict_x(x) for x in X]
        return np.array(predictions)

    # Predict label for x
    def __predict_x(self, x):
        # distances between x and all examples
        distances = (euclidean_distance(x, example) for example in self.X)

        # Sort all examples by their distance to x.
        neighbors = sorted(((dist, target)
                            for (dist, target) in zip(distances, self.y)),
                           key=lambda x: x[0])

        neighbors_targets = [target for (_, target) in neighbors[:self.k]]

        return self.get_most_common(neighbors_targets)
