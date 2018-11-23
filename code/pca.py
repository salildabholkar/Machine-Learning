from scipy.linalg import svd
import numpy

from utils.CommonSetup import CommonSetup


class PCA(CommonSetup):
    y_required = False

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, data, y=None):
        self.mean = numpy.mean(data, axis=0)
        X = data.copy()
        X -= self.mean
        w, v = numpy.linalg.eig(numpy.cov(X.T))
        v_t = v.T

        variance_ratio = w ** 2 / (w ** 2).sum()
        print('Variances: ', (variance_ratio[0:self.n_components]))
        self.components = v_t[0:self.n_components]

    def transform(self, X):
        X = X.copy()
        X -= self.mean
        return numpy.dot(X, self.components.T)

    def _predict(self, X=None):
        return self.transform(X)
