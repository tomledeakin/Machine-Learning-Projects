import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


with np.load('mnist.npz') as f:
    x_train = f["x_train"]
    x_test = f["x_test"]
    y_train = f["y_train"]
    y_test = f["y_test"]
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)

cls = DecisionTreeClassifier()
cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)
print(classification_report(y_test, y_predict))


