from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from code.logistic_regression import LogisticRegression
from utils.helpers import mean_squared_error, accuracy


def logistic_regression_init():
    X, y = make_classification(n_samples=5000,
                               n_features=500,
                               n_classes=2,
                               class_sep=2.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f'[Logistic Regression] Accuracy (with {X.shape[0]} samples):', accuracy(y_test, predictions))
    print(f'[Logistic Regression] Error (with {X.shape[0]} samples):', mean_squared_error(y_test, predictions))


if __name__ == '__main__':
    logistic_regression_init()
