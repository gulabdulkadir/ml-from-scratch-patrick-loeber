import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from PCA import PCA
from Decision_Tree import DecisionTree

iris = datasets.load_iris()
X, y = iris.data, iris.target
pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

dc = DecisionTree()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=83)
dc.fit(X_train,  y_train)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

predictions = dc.predict(X_test)
print ("Decision tree acuracy with PCA: ", accuracy(y_test, predictions))