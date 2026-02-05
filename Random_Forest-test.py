import numpy as np
from Random_Forest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn import datasets

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=83)

random_forest = RandomForest()
random_forest.fit(X_train, y_train)
predicts = random_forest.predict(X_test)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

print ("random forest accuracy: ", accuracy(y_test, predicts))