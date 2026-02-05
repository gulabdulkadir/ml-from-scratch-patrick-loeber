import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from Decision_Tree import DecisionTree
import time
t1 = time.time()
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=83)

def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc

decision_tree = DecisionTree()
decision_tree.fit(X_train, y_train)
predictions = decision_tree.predict(X_test)
t2 = time.time()
print("Decision Tree accuracy: ", accuracy(y_test, predictions), "Çalışma süresi: ", t2 - t1)

