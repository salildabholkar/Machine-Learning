from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

from code.linear_regression import LinearRegression
from utils.helpers import mean_squared_error, accuracy


def linear_regression_init():
    # Generate a random regression problem
    X, y = make_regression(n_samples=10000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('[Linear Regression] MSE (with 10000 samples):', mean_squared_error(y_test, predictions))


if __name__ == '__main__':
    linear_regression_init()
