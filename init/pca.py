from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from code.logistic_regression import LogisticRegression
from utils.helpers import accuracy
from code.pca import PCA


def pca_init():
    X, y = make_classification(n_samples=5000, n_features=100, n_informative=75, class_sep=2.5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    pca = PCA(5)

    pca.fit(X_train)
    X_train_reduced = pca.transform(X_train)
    X_test_reduced = pca.transform(X_test)

    model = LogisticRegression(lr=0.001, max_iters=2500)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f'[PCA] Logistic regression {X_train.shape} before reduction:', accuracy(y_test, predictions))

    model.fit(X_train_reduced, y_train)
    predictions = model.predict(X_test_reduced)
    print(f'[PCA] Logistic regression {X_train_reduced.shape} after reduction:', accuracy(y_test, predictions))

if __name__ == '__main__':
    pca_init()
