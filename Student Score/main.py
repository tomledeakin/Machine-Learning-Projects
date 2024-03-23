import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lazypredict.Supervised import LazyRegressor, LazyClassifier




data = pd.read_csv('StudentScore.xls')
print(data.info())
# profile = ProfileReport(data, title='Report', explorative=True)
# profile.to_file('student_report.html')

target = 'math score'
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)

# Numerical data preprocessing --------------------------------------------------------------------------------------------------------
num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                  ('scaler', StandardScaler())
])

# result = num_transformer.fit_transform(x_train[["reading score", "writing score"]])
#
# for i, j in zip(x_train[["reading score", "writing score"]].values, result):
#     print("Before: {}. After: {}".format(i, j))

# Ordinal data preprocessing --------------------------------------------------------------------------------------------------------
# print(data["parental level of education"].unique())

education_levels = ["master's degree", "bachelor's degree", "associate's degree", "some college", "high school", "some high school"]
gender_values = x_train["gender"].unique().tolist()
lunch_values = x_train["lunch"].unique().tolist()
test_prep_values = x_train["test preparation course"].unique().tolist()




ord_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy="most_frequent")),
                                  ("ordinalencoder", OrdinalEncoder(categories=[education_levels, gender_values, lunch_values, test_prep_values]))])

# result = ord_transformer.fit_transform(x_train[["parental level of education", "gender", "lunch", "test preparation course"]])
#
# for i, j in zip(x_train[["parental level of education", "gender", "lunch", "test preparation course"]].values, result):
#     print("Before: {}. After: {}".format(i, j))

# Nominal data preprocessing --------------------------------------------------------------------------------------------------------
nom_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy="most_frequent")),
                                  ('encoder', OneHotEncoder(sparse_output=False))])

# result = nom_transformer.fit_transform(x_train[["race/ethnicity"]])
#
# for i, j in zip(x_train[["race/ethnicity"]].values, result):
#     print("Before: {}. After: {}".format(i, j))


preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("ord_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_features", nom_transformer, ["race/ethnicity"])]
)

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor())
])

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)



# param_grid = {
#     'model__n_estimators': [50, 100, 200],
#     'model__criterion': ['squared_error', 'absolute_error', 'poisson'],
#     'model__max_depth': [None, 2, 5, 10],
#     'model__max_features': [1.0, "sqrt", "log2"],
#     'preprocessor__num_features__imputer__strategy': ['mean', 'median']
# }
# grid_search = RandomizedSearchCV(reg, param_distributions=param_grid, cv=5, verbose=2, scoring='r2', n_jobs=-1, n_iter=30)
# grid_search.fit(x_train, y_train)
# print(grid_search.best_params_)
# y_predict = grid_search.predict(x_test)
# mse = mean_squared_error(y_test, y_predict)
# mae = mean_absolute_error(y_test, y_predict)
# r2 = r2_score(y_test, y_predict)
# print("MSE: ", mse)
# print("MAE: ", mae)
# print("R2: ", r2)

# MSE:  28.00070079877387
# MAE:  4.189583645430788
# R2:  0.8507073386635315


clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, prediction = clf.fit(x_train, x_test, y_train, y_test)


