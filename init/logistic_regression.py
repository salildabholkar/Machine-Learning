from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from code.logistic_regression import LogisticRegression
from utils.helpers import mean_squared_error, accuracy


def logistic_regression_init():
    X, y = make_classification(n_samples=1000,
                               n_features=100,
                               n_classes=2,
                               class_sep=2.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('[Logistic Regression] Accuracy (with 10000 samples):', accuracy(y_test, predictions))
    print('[Logistic Regression] Error (with 10000 samples):', mean_squared_error(y_test, predictions))


if __name__ == '__main__':
    logistic_regression_init()
