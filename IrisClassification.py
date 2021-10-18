from __future__ import print_function
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('Labels :', np.unique(iris_y))

# weights function self defined give more accuracy
def myWeight(distances):
    sigma2 = .5
    return np.exp(-distances**2/sigma2)

#np.random.seed(7)
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=30)

model = neighbors.KNeighborsClassifier(n_neighbors=100, p=2, weights=myWeight)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy of 100NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
# K nearest neighbors is sensitive in case of small K clustering.
