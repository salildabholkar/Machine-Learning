from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error

from code.linear_regression import LinearRegression
from utils.helpers import mean_squared_error


def linear_regression_init():
    X, y = make_regression(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('[Linear Regression] R^2 Accuracy (with 10000 samples):', r2_score(y_test, predictions))
    print('[Linear Regression] MSE (with 10000 samples):', mean_squared_error(y_test, predictions))


if __name__ == '__main__':
    linear_regression_init()
