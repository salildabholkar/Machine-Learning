from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from code.naive_bayes import NaiveBayes


def naive_bayes_init():
    # random binary classification.
    X, y = make_classification(n_samples=1000, n_classes=2, class_sep=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = NaiveBayes()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)[:, 1]

    print('[Naive Bayes] Accuracy (1000 samples):', roc_auc_score(y_test, predictions))


if __name__ == '__main__':
    naive_bayes_init()
