# Importing Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_validate
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

data = sns.load_dataset('titanic')
df = data.copy()

# Data Preparation

df.head()
df.info()
df.describe().T
df.isnull().sum()

# Outlier Check

def outlier_threshold(dataframe, col_name, quantile1 = 0.01, quantile3 = 0.99):
    q1 = dataframe[col_name].quantile(quantile1)
    q3 = dataframe[col_name].quantile(quantile3)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound

df.head()

low, up = outlier_threshold(df, "fare")

df[(df["fare"] < low) | (df["fare"] > up)].head()

outlier_threshold(df, "age")

# Checking Categorical Columns and Numerical Columns
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

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols

df.head()

# Missing Values Checking

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])
df.loc[(df["age"].isnull()) & (df["sex"] == "female"), "age"] = df.groupby("sex")["age"].mean()["female"]
df.loc[(df["age"].isnull()) & (df["sex"] == "male"), "age"] = df.groupby("sex")["age"].mean()["male"]

df["deck"].value_counts()

df["deck"] = df["deck"].fillna(df["deck"].mode()[0])

missing_values_table(df)

# Encoding

df.head()
df.drop("alive", axis = 1,inplace=True)

## Label
le = LabelEncoder()
df["survived"] = le.fit_transform(df["survived"])
df["who"] = le.fit_transform(df["who"])
df["sex"] = le.fit_transform(df["sex"])
df["adult_male"] = le.fit_transform(df["adult_male"])
df["alive"] = le.fit_transform(df["alive"])
df["alone"] = le.fit_transform(df["alone"])

## One-Hot
df = pd.get_dummies(df, columns = ["embarked"], drop_first = True)
df.head()
df = pd.get_dummies(df, columns = ["pclass"], drop_first = True)
df.head()
df = pd.get_dummies(df, columns = ["class"], drop_first = True)
df.head()
df = pd.get_dummies(df, columns = ["embark_town"], drop_first = True)
df.head()
df = pd.get_dummies(df, columns = ["deck"], drop_first = True)
df.head()

# Scaling
ss = StandardScaler()
df[["age","fare"]] = ss.fit_transform(df[["age","fare"]])

df.head()

# Modeling

X = df.drop("survived", axis = 1)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

## RandomForestClassifier
rf_model = RandomForestClassifier(random_state=23)
rf_model.fit(X_train, y_train)

### Before Hyperparameter Optimization
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
accuracy

### Finding Best Parameters
rf_params = {"max_depth": [5, 8,10, None],
             "max_features": [3, 5, "auto"],
             "min_samples_split": [2, 5, 8,10],
             "n_estimators": [200, 300, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state = 23).fit(X, y)

### After Hyperparameter Optimization
y_pred = rf_final.predict(X_test)
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
accuracy

## LightGBM

lgb_model = LGBMClassifier(random_state=23)
lgb_model.fit(X_train, y_train)

### Before Hyperparameter Optimization
y_pred = lgb_model.predict(X_test)
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
accuracy

### Finding Best Parameters
lgb_model.get_params()
lgb_params = {"learning_rate" : [0.01,0.03,0.05,0.07],
              "n_estimators" : [200,500,1000,2000],
              "max_depth" : [3,5,7,9,None]}

lgb_best_grid = GridSearchCV(lgb_model, lgb_params, cv=5, n_jobs=-1).fit(X, y)
lgb_best_grid.best_params_

lgb_final = lgb_model.set_params(**lgb_best_grid.best_params_).fit(X,y)

### After Hyperparameter Optimization
y_pred = lgb_final.predict(X_test)
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
accuracy