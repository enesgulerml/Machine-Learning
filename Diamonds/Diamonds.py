# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report


pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)

data = sns.load_dataset("diamonds")
df = data.copy()

# Data Preparation
df.head()
df.isnull().sum()
df.describe().T
df.info()

# Checking Outliers
def outlier_thresholds(dataframe, col_name, quantile1=0.01, quantile3=0.99):
    q1 = dataframe[col_name].quantile(quantile1)
    q3 = dataframe[col_name].quantile(quantile3)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols

df.head()

for col in num_cols:
    print(col, check_outlier(df, col))

t_lower, t_upper = outlier_thresholds(df, "table")
df[(df["table"] < t_lower) | (df["table"] > t_upper)]
median_table = df['table'].median()
df.loc[(df["table"] < t_lower) | (df["table"] > t_upper), 'table'] = median_table

d_lower, d_upper = outlier_thresholds(df, "depth")
df[(df["depth"] < d_lower) | (df["depth"] > d_upper)]
mean_depth = df['depth'].mean()
df.loc[(df["depth"] < d_lower) | (df["depth"] > d_upper), 'depth'] = mean_depth

y_lower, y_upper = outlier_thresholds(df, "y")
df[(df["y"] < y_lower) | (df["y"] > y_upper)]
mean_y = df['y'].mean()
df.loc[(df["y"] < y_lower) | (df["y"] > y_upper), 'y'] = mean_y

z_lower, z_upper = outlier_thresholds(df, "z")
df[(df["z"] < z_lower) | (df["z"] > z_upper)]
mean_z = df['z'].mean()
df.loc[(df["z"] < z_lower) | (df["z"] > z_upper), 'z'] = mean_z

for col in num_cols:
    print(col, check_outlier(df, col))

# Checking Missing Values
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# Encoding

## Label
encoder = LabelEncoder()
df['cut_encoded'] = encoder.fit_transform(df['cut'])
df['color_encoded'] = encoder.fit_transform(df['color'])
df['clarity_encoded'] = encoder.fit_transform(df['clarity'])
df = df.drop(['cut', 'color', 'clarity'], axis=1)
df.head()

# Modeling

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

dt_model = DecisionTreeRegressor(random_state=23)
dt_model.fit(X_train, y_train)

### Before Hyperparameter Optimization
y_pred = dt_model.predict(X_test)

dt_mae = mean_absolute_error(y_test, y_pred)
dt_mse = mean_squared_error(y_test, y_pred)
dt_r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {dt_mae}")
print(f"Mean Squared Error: {dt_mse}")
print(f"R2 Score: {dt_r2}")

## Hyperparameter Optimization
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_dt.fit(X_train, y_train)
grid_search_dt.best_params_
final_dt = dt_model.set_params(**grid_search_dt.best_params_, random_state=23).fit(X,y)

### After Hyperparameter Optimization
y_pred = final_dt.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred)
rf_mse = mean_squared_error(y_test, y_pred)
rf_r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {rf_mae}")
print(f"Mean Squared Error: {rf_mse}")
print(f"R2 Score: {rf_r2}")

## Random Forest

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

### Before Hyperparameter Optimization
y_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred)
rf_mse = mean_squared_error(y_test, y_pred)
rf_r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {rf_mae}")
print(f"Mean Squared Error: {rf_mse}")
print(f"R2 Score: {rf_r2}")

## Hyperparameter Optimization
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)
grid_search_rf.best_params_
rf_final = rf_model.set_params(**grid_search_rf.best_params_, random_state = 23).fit(X,y)

### After Hyperparameter Optimization
y_pred = rf_final.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred)
rf_mse = mean_squared_error(y_test, y_pred)
rf_r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {rf_mae}")
print(f"Mean Squared Error: {rf_mse}")
print(f"R2 Score: {rf_r2}")

