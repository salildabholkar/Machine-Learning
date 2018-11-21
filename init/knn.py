from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from code import knn
from utils.helpers import accuracy


def knn_init():
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=5,
                               n_redundant=0, n_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    knn_obj = knn.KNN(k=5)

    knn_obj.fit(X_train, y_train)
    predictions = knn_obj.predict(X_test)
    print('[KNN] Classification Accuracy (1000 samples):', accuracy(y_test, predictions))


if __name__ == '__main__':
    knn_init()
