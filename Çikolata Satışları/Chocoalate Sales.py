import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
# !pip install optuna
import optuna
warnings.filterwarnings("ignore")


from datashader.colors import viridis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.options.display.float_format = '{:,.3f}'.format


df = pd.read_csv("Projects/ML/Çikolata Satışları/Chocolate Sales.csv")

df.head()

df.isnull().sum()

df.info()

df.shape

df.describe().T

def check_values(dataframe, variable):
    print(dataframe[variable].value_counts())

check_values(df, "Country")


sns.barplot(x = "Boxes Shipped", y = "Country",color = "lightcoral", data = df)
plt.show()

dff = pd.DataFrame(df[["Sales Person","Country","Product"]])

for col in dff.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=df[col], palette="viridis")
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x="Sales Person", y="Boxes Shipped", data=df, palette="flare")
plt.title("Shipped Units per Representative")
plt.xlabel("Representative")
plt.ylabel("Shipped Units")
plt.xticks(rotation=45)
plt.show()

#### Encoding

le = LabelEncoder()

df['Sales Person'] = le.fit_transform(df['Sales Person'])
df['Country'] = le.fit_transform(df['Country'])
df['Product'] = le.fit_transform(df['Product'])

df.head()

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df = df.drop('Date',axis=1)

df['Amount'] = df['Amount'].replace({'\$': '',',':''}, regex=True).astype(int)

df.head()

plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap="rocket", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix", fontsize=12)
plt.show()

######### Machine learning

y = df['Amount']
X = df.drop('Amount',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 23)


def get_params(trial):
    model_type = trial.suggest_categorical("model", ["lightgbm", "randomforest", "ridge"])

    if model_type == "lightgbm":
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "n_jobs": -1  # Use all CPU cores
        }
        model = lgb.LGBMRegressor(**params)

    elif model_type == "randomforest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "n_jobs": -1  # Use all CPU cores
        }
        model = RandomForestRegressor(**params)

    elif model_type == "ridge":
        params = {
            "alpha": trial.suggest_float("alpha", 0.1, 100.0, log=True),
        }
        model = Ridge(**params)

    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    return np.mean(score)


study = optuna.create_study(direction="maximize")
study.optimize(get_params, n_trials=50)

print("Best Model:", study.best_trial.params["model"])
print("Best Parameters:", study.best_trial.params)

best_model_params = study.best_trial.params.copy()
best_model = None

if best_model_params["model"] == "lightgbm":
    best_model_params.pop("model")
    best_model = lgb.LGBMRegressor(**best_model_params)

elif best_model_params["model"] == "randomforest":
    best_model_params.pop("model")
    best_model = RandomForestRegressor(**best_model_params)

elif best_model_params["model"] == "ridge":
    best_model_params.pop("model")
    best_model = Ridge(**best_model_params)

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
final_mse = mean_squared_error(y_test, y_pred)
final_mse

y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
r2