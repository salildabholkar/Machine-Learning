import random
import numpy as np

from utils.CommonSetup import CommonSetup
from utils.helpers import euclidean_distance

random.seed(2)


class KMeans(CommonSetup):
    y_required = False

    # K : The number of clusters into which the dataset is partitioned.
    def __init__(self, K = 5):
        print('Starting KMeans algorithm')
        self.K = K
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def _predict(self, X = None):
        self.centroids = [self.X[x] for x in random.sample(range(self.n_samples), self.K)]
        for _ in range(100):  # Max 100 iterations allowed if not converged
            self.__assign_centroids(self.centroids)
            get_centroid = lambda cluster: [np.mean(np.take(self.X[:, i], cluster)) for i in range(self.n_features)]
            new_centroids = [get_centroid(cluster) for cluster in self.clusters]

            if self.__is_converged(self.centroids, new_centroids):
                break

        self.centroids = new_centroids

        return self.__make_predictions()

    def __make_predictions(self):
        predictions = np.empty(self.n_samples)

        for i, cluster in enumerate(self.clusters):
            for index in cluster:
                predictions[index] = i
        return predictions

    def __assign_centroids(self, centroids):

        for row in range(self.n_samples):
            for i, cluster in enumerate(self.clusters):
                if row in cluster:
                    self.clusters[i].remove(row)
                    break

            closest = self.__find_closest_centroid(row, centroids)
            self.clusters[closest].append(row)

    def __find_closest_centroid(self, fpoint, centroids):  # closest centroid for a point
        closest_index = None
        closest_distance = None
        for i, point in enumerate(centroids):
            dist = euclidean_distance(self.X[fpoint], point)
            if closest_index is None or dist < closest_distance:
                closest_index = i
                closest_distance = dist
        return closest_index

    def __find_next_center(self):
        distances = np.array([min([euclidean_distance(x, c) for c in self.centroids]) for x in self.X])
        probs = distances / distances.sum()
        cumprobs = probs.cumsum()
        r = random.random()
        ind = np.where(cumprobs >= r)[0][0]
        return self.X[ind]

    def __is_converged(self, centroids_old, centroids): # If prev and new are same
        distance = 0
        for i in range(self.K):
            distance += euclidean_distance(centroids_old[i], centroids[i])
        return distance == 0
