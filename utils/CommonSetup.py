import numpy as np


class CommonSetup(object):

    # Initialize defaults
    # X : Feature dataset.
    # y : Target values.

    X = None
    y = None
    y_required = True
    fit_required = True

    def _fit_model(self, X, y = None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError('No feature added')

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError('y is required')

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError('No target values')

        self.y = y

    def fit(self, X, y=None):
        self._fit_model(X, y)

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError('Model has not been fit')

    def _predict(self, X=None):  # must be implemented in child class
        raise NotImplementedError()
