import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
# from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
import re


def filter_location(string):
    result = re.findall("\,\ [A-Z]{2}$", string)
    if len(result) == 1:
        return result[0][-2:]
    else:
        return string


data = pd.read_excel("final_project.ods", engine="odf", dtype=str)
data["location"] = data["location"].apply(filter_location)
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2024, stratify=y)
# print(y_train.value_counts())
# ros = SMOTEN(random_state=42, k_neighbors=2, sampling_strategy={
#     "bereichsleiter": 1000,
#     "director_business_unit_leader": 500,
#     "specialist": 500,
#     "managing_director_small_medium_company": 500,
# })
# print("---------------------------------")
# x_train, y_train = ros.fit_resample(x_train, y_train)
# print(y_train.value_counts())

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry"),
])

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    # ("feature_selection", SelectKBest(chi2, k=200)),
    ("feature_selection", SelectPercentile(chi2, percentile=5)),
    ("model", RandomForestClassifier())
])

# processed_data = cls.fit_transform(x_train)
# print(processed_data.shape)
# x_train = cls.fit(x_train, y_train)
# y_predict = cls.predict(x_test)
# print(classification_report(y_test, y_predict))
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__criterion": ["gini", "entropy", "log_loss"],
    "feature_selection__percentile": [1, 5, 10],
    "preprocessor__description__ngram_range": [(1, 1), (1, 2), (2, 2)]
}
grid_search = RandomizedSearchCV(cls, param_distributions=param_grid, cv=5, scoring="f1_weighted", n_jobs=6, verbose=2,
                                 n_iter=2)
x_train = grid_search.fit(x_train, y_train)
#
# filename = 'best_model.pkl'
# pickle.dump(grid_search, open(filename, 'wb'))


# model = pickle.load(open(filename, 'rb'))
y_predict = grid_search.predict(x_test)
print(classification_report(y_test, y_predict))
