import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from AdaBoost import AdaBoost

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_pred)

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=83)
AdaBoost = AdaBoost(n_clf=5)
AdaBoost.fit(X_train, y_train)
predictions = AdaBoost.predict(X_test)

print("AdaBoost accuracy: ", accuracy(y_test, predictions))
