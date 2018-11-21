import numpy as np
from sklearn.datasets import make_blobs

from code.Kmeans import KMeans


def kmeans_init():
    X, y = make_blobs(centers = 4, n_samples = 5, n_features=2,
                      shuffle = True, random_state = 42)

    print('Initial Data:')
    print(X, '\n')
    clusters = len(np.unique(y))
    k = KMeans(K = clusters)
    k.fit(X)
    k.predict()
    print('Arranged in following ', clusters, ' clusters by index:')
    print(k.clusters)


if __name__ == '__main__':
    kmeans_init()
