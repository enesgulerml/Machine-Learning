import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

data = sns.load_dataset('tips')
df = data.copy()

## Data Preparation

df.head()
df.isnull().sum()
df.describe().T
df.nunique()

df["total_bill"] = df["total_bill"] - df["tip"]

df.head()

def outlier_thresholds(dataframe, variable, q1 = 0.25, q3 = 0.75):
    quantile1 = dataframe[variable].quantile(q1)
    quantile3 = dataframe[variable].quantile(q3)
    iqr = quantile3 - quantile1
    up = quantile3 + 1.5 * iqr
    down = quantile1 - 1.5 * iqr
    return up, down

outlier_thresholds(df,"total_bill")
outlier_thresholds(df,"tip")

down, up = outlier_thresholds(df,"total_bill")

df[(df["total_bill"] > up) | (df["total_bill"] < down)].index

def get_col_names(dataframe, cat_th=10, car_th=20):
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

cat_cols, num_cols, cat_but_car = get_col_names(df)

cat_cols
num_cols

# New Features

df["Daily_Purchase"] = df["total_bill"] / 4
df["Daily_Tip"] = df["tip"] / 4

# Encoding

## One-Hot
df.head()

df = pd.get_dummies(df, columns=["day"])
df = pd.get_dummies(df, columns = ["time"])
df = pd.get_dummies(df, columns = ["smoker"])

df.head()

## LabelEncoder
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df.head()

# Scaling

ss = StandardScaler()
df["total_bill"] = ss.fit_transform(df[["total_bill"]])
df["tip"] = ss.fit_transform(df[["tip"]])
df["Daily_Purchase"] = ss.fit_transform(df[["Daily_Purchase"]])
df["Daily_Tip"] = ss.fit_transform(df[["Daily_Tip"]])
df.head()

# Modeling

## RandomForest
X = df.drop("total_bill", axis = 1)
y = df["total_bill"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

rf_model = RandomForestRegressor(random_state=23)

rf_model.fit(X_train, y_train)

rf_model.get_params()

y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse
rmse
mae
r2

rf_model.get_params()

rf_params = {"max_depth": [3,5,8,None],
             "max_features": [3,5,8,11,"auto"],
             "min_samples_split": [2,5,8,10,11],
             "n_estimators": [200, 300, 500,1000,2000]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state = 23).fit(X, y)

y_pred = rf_final.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse
rmse
mae
r2

##LGBM

model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

model.get_params()

lgbm_params = {"learning_rate" : [0.01,0.03,0.05,0.07,0.09],
               "max_depth" : [1,3,5,7,9],
               "n_estimators" : [200,300,500,1000,2000]}

lgb_best_grid = GridSearchCV(model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgb_best_grid.best_params_

lgb_final = model.set_params(**lgb_best_grid.best_params_, random_state = 23).fit(X,y)

y_pred = lgb_final.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse
rmse
mae
r2

# XGB Regressor

xgb_model = XGBRegressor(random_state=23)
xgb_model.fit(X_train, y_train)
xgb_model.get_params()

y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse
rmse
mae
r2
