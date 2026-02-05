import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from Perceptron import Perceptron

def accuracy(y_true, y_predict):
    return (np.sum(y_predict == y_true) / len(y_true))

X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=83)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=83)

p = Perceptron(lr=0.01, n_iters=1000)
p.fit(X_train, y_train)
pred = p.predict(X_test)

print ("Perceptron accuracy", accuracy(y_test, pred))