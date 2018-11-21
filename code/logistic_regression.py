from autograd import numpy, grad

from utils.CommonSetup import CommonSetup
from utils.helpers import binary_crossentropy

numpy.random.seed(5)


class LogisticRegression(CommonSetup):
    def __init__(self, lr=0.01, max_iters=2000, C=0.03, tolerance=0.0001):
        self.C = C
        self.tolerance = tolerance
        self.lr = lr
        self.max_iters = max_iters
        self.errors = []
        self.theta = []
        self.n_samples, self.n_features = None, None

    def __bc_loss(self, w):
        sigmoid = lambda x: 0.5 * (numpy.tanh(x) + 1)
        loss = binary_crossentropy(self.y, sigmoid(numpy.dot(self.X, w)))
        return self.__with_penalty(loss, w)

    def __with_penalty(self, loss, w):
        loss += (0.5 * self.C) * (w[1:] ** 2).sum() ## L2 loss
        return loss

    def __cost(self, X, y, theta):
        prediction = X.dot(theta)
        error = binary_crossentropy(y, prediction)
        return error

    def fit(self, X, y=None):
        self._fit_model(X, y)
        self.n_samples, self.n_features = X.shape

        # weights + bias
        self.theta = numpy.random.normal(size=(self.n_features + 1), scale=0.5)
        self.X = self.__add_intercept(self.X)

        self.theta, self.errors = self.__gradient_descent()

    def __add_intercept(self, X):
        b = numpy.ones([X.shape[0], 1])
        return numpy.concatenate([b, X], axis=1)

    def _predict(self, X=None):
        X = self.__add_intercept(X)
        sigmoid = lambda x: 0.5 * (numpy.tanh(x) + 1)
        return sigmoid(X.dot(self.theta))

    def __gradient_descent(self):
        theta = self.theta
        errors = [self.__cost(self.X, self.y, theta)]
        # derivative of the loss
        cost_d = grad(self.__bc_loss)
        for i in range(1, self.max_iters + 1):
            delta = cost_d(theta)
            theta -= self.lr * delta

            errors.append(self.__cost(self.X, self.y, theta))

            error_diff = numpy.linalg.norm(errors[i - 1] - errors[i])
            if error_diff < self.tolerance:
                break
        return theta, errors
