import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# DATA PREPARATION

data = sns.load_dataset('iris')
df = data.copy()

def check_dataframe(dataframe):
    print('Checking dataframe...')
    print("########## SHAPE ##########")
    print(dataframe.shape)
    print("########## NA ##########")
    print(dataframe.isnull().sum())
    print("########## INFO ##########")
    print(dataframe.info())
    print("########## HEAD ##########")
    print(dataframe.head())
    print("########## TAIL ##########")
    print(dataframe.tail())
    print("########## UNIQUE ##########")
    print(dataframe.nunique())

check_dataframe(df)

# CHECKING OUTLIERS

def outlier_thresholds(dataframe, variable, q1=0.01, q3=0.99):
    q1 = dataframe[variable].quantile(q1)
    q3 = dataframe[variable].quantile(q3)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound

def replace_with_thresholds(dataframe, variable):
    lower_bound, upper_bound = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < lower_bound, variable] = lower_bound
    dataframe.loc[dataframe[variable] > upper_bound, variable] = upper_bound

def check_outliers(dataframe, variable):
    lower_bound, upper_bound = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < lower_bound) | (dataframe[variable] > upper_bound)].any(axis=None):
        return True
    else:
        return False

## With seaborn
sns.boxplot(x=df["sepal_length"])
plt.show()
sns.boxplot(x=df["sepal_width"])
plt.show()
# This variable appears to have some outliers. But that could also mean that these values are not outliers, but rather extreme values!
sns.boxplot(x=df["petal_length"])
plt.show()
sns.boxplot(x=df["petal_width"])
plt.show()

lower_sw, upper_sw = outlier_thresholds(df, 'sepal_width')

check_outliers(df, 'sepal_width')
# And here we see these values are not outliers. They are extreme values.

# Getting Col Names

def get_col_names(dataframe, categorical_th = 10, cardinal_th = 20):

    # Cardinal and Categorical Columns
    categorical_columns = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    numerical_but_categorical_columns = [col for col in dataframe.columns if dataframe[col].nunique() < categorical_th and
                                         dataframe[col].dtypes != "O"]
    categorical_but_cardinal_columns = [col for col in dataframe.columns if dataframe[col].nunique() > cardinal_th and
                                        dataframe[col].dtypes == "O"]
    categorical_columns = categorical_columns + numerical_but_categorical_columns
    categorical_columns = [col for col in categorical_columns if col not in categorical_but_cardinal_columns]

    # Numerical Columns
    numerical_columns = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    numerical_columns = [col for col in dataframe.columns if col not in numerical_but_categorical_columns]

    print(f'Categorical Columns: {len(categorical_columns)}')
    print(f'Numerical Columns: {len(numerical_columns)}')
    print(f'Categorical But Cardinal Columns: {len(categorical_but_cardinal_columns)}')
    print(f'Numerical But Cardinal Columns: {len(numerical_but_categorical_columns)}')
    return categorical_columns, numerical_columns, categorical_but_cardinal_columns

cat_cols, num_cols, cat_but_car = get_col_names(df)

num_cols = [col for col in num_cols if col != "species"]

# CHECKING MISSING VALUES

def missing_values_table(dataframe):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)
    missing_dataframe = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_dataframe, end="\n")

missing_values_table(df)

# Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols, drop_first=False)
df.head()

# Modeling
## For Versicolor
X = df.drop(["species_versicolor","species_virginica","species_setosa"], axis=1)
y = df["species_versicolor"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

xgb_model = XGBClassifier(random_state = 23, use_label_encoder=False)
xgb_model.fit(X_train, y_train)
cv_results = cross_validate(xgb_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgb_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgb_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

## For Virginica
X = df.drop(["species_versicolor","species_virginica","species_setosa"], axis=1)
y = df["species_virginica"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

xgb_model = XGBClassifier(random_state = 23, use_label_encoder=False)
xgb_model.fit(X_train, y_train)
cv_results = cross_validate(xgb_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgb_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgb_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

## Setosa
X = df.drop(["species_versicolor","species_virginica","species_setosa"], axis=1)
y = df["species_setosa"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

xgb_model = XGBClassifier(random_state = 23, use_label_encoder=False)
xgb_model.fit(X_train, y_train)
cv_results = cross_validate(xgb_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgb_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgb_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(xgb_model, X_train)