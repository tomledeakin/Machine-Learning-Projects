import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_csv('diabetes.csv')

corr = data.corr()

# profile = ProfileReport(data, title='Report', explorative=True)
# profile.to_file('diabetes_report.html')

target = 'Outcome'
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))
# print(len(x_val), len(y_val))

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'criterion': ['gini', 'entropy', 'log_loss'],
#     'max_depth': [None, 2, 5, 10]
# }
# grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=4, verbose=2, scoring='f1')

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, prediction = clf.fit(x_train, x_test, y_train, y_test)
print(models)

# clf = RandomForestClassifier(n_estimators=50, criterion="entropy", random_state=33)
# grid_search.fit(x_train, y_train)
#
# print(grid_search.best_params_)
# print('score of validation')
# print(grid_search.best_score_)
#
# y_predict = grid_search.predict(x_test)
# print(y_predict.shape)
# print(y_test.shape)

# count = 0
# total = 0
# for i, j in zip(y_predict, y_test):
#     print(f"Predicted: {i}, Actual: {j}")
#
# for i, j in zip(y_predict, y_test):
#     total += 1
#     if i == j:
#         count += 1
# print(f"accuracy: {100 * count/total}%")

print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))


# cm = np.array(confusion_matrix(y_test, y_predict, labels=[0,1]))
# dataframe = pd.DataFrame(cm, index=["Not diabetic", "diabetic"], columns=["Predicted Healthy", "Predicted diabetic"])
# sns.heatmap(dataframe, annot=True)
# plt.savefig("diabetes.png")