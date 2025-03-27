# Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

# Data Preparation
data = pd.read_csv("Projects/ML_Project/CO_2 Project/co2.csv")
df = data.copy()

df.head()
df.info()
df.describe().T
df.isnull().sum()

df.rename(columns={'Make':'make',
                  'Model':'model',
                  'Vehicle Class': 'vehicle_class',
                  'Engine Size(L)': 'engine_size',
                  'Cylinders': 'cylinders',
                  'Transmission':'transmission',
                  'Fuel Type': 'fuel_type',
                  'Fuel Consumption Comb (L/100 km)':'fuel_cons_comb',
                  'Fuel Consumption Comb (mpg)':'fuel_cons_comb_mpg',
                  'Fuel Consumption Hwy (L/100 km)': 'fuel_cons_hwy',
                  'CO2 Emissions(g/km)': 'co2',
                  'Fuel Consumption City (L/100 km)':'fuel_cons_city'
                  }, inplace=True)

# Data Visualization
fig = plt.figure(figsize=(10, 5))
counts = df.make.value_counts().sort_values(ascending=False).head(20)

counts.plot(kind="bar", color="royalblue", edgecolor="black", alpha=0.7)
plt.ylabel("Car Count", fontsize=12)
plt.xlabel("Car Brands", fontsize=12)
plt.title("Top 20 Car Brands by Count", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))
sns.boxplot(x="make", y="co2", data=df, palette="coolwarm", fliersize=3, linewidth=1.2)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Car Brands", fontsize=14, fontweight="bold")
plt.ylabel("CO2 Emissions", fontsize=14, fontweight="bold")
plt.title("CO2 Emissions by Car Brands", fontsize=16, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.box(False)
plt.show()

fig = plt.figure(figsize=(12, 6))
counts = df.model.value_counts().sort_values(ascending=False).head(20)
counts.plot(kind="bar", color="darkorange", edgecolor="black", alpha=0.8)
plt.ylabel("Car Count", fontsize=12, fontweight="bold")
plt.xlabel("Car Models", fontsize=12, fontweight="bold")
plt.title("Top 20 Car Models by Count", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# LinearRegression

X = df[["engine_size"]]
y = df["co2"]

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

lr = LinearRegression()
lr.fit(X_train, y_train)

lr.coef_[0]
lr.intercept_

# Predicting Test Data

y_pred_tr = lr.predict(X_train)
y_pred_te = lr.predict(X_test)

plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred_te, ci=None, scatter_kws={"alpha": 0.6, "color": "dodgerblue"}, line_kws={"color": "red", "linewidth": 2})
plt.xlabel("Actual Values (y_test)", fontsize=12, fontweight="bold")
plt.ylabel("Predicted Values (y_pred)", fontsize=12, fontweight="bold")
plt.title("Actual vs. Predicted Values", fontsize=14, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

def train_val(y_train, y_train_pred, y_test, y_test_pred, model_name):
    scores = {
        model_name + "_train_score": {
            "R²": r2_score(y_train, y_train_pred),
            "MAE": mean_absolute_error(y_train, y_train_pred),
            "MSE": mean_squared_error(y_train, y_train_pred),
            "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred))
        },
        model_name + "_test_score": {
            "R²": r2_score(y_test, y_test_pred),
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "MSE": mean_squared_error(y_test, y_test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred))
        }
    }
    return pd.DataFrame(scores)

slr_score = train_val(y_train, y_pred_tr, y_test, y_pred_te, "s_linear")
slr_score

##
X = df[["engine_size","fuel_cons_city","fuel_cons_hwy","fuel_cons_comb"]]
y = df["co2"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

model = LinearRegression()
model.fit(X_train, y_train)

model.coef_[0]
model.intercept_


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_val(y_train,y_train_pred, y_test, y_test_pred, "linear")

## Cross-validation

model = LinearRegression()
scores = cross_validate(model, X_train, y_train,
                       scoring=["r2","neg_mean_absolute_error","neg_root_mean_squared_error"], cv=5)

print("Mean R² Score:", np.mean(scores["test_r2"]))
print("Mean MAE Score:", -np.mean(scores["test_neg_mean_absolute_error"]))
print("Mean RMSE Score:", -np.mean(scores["test_neg_root_mean_squared_error"]))

model_r = Ridge()
param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(
    model_r, param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("🔍 Best Parameters:", grid_search.best_params_)
print(f"📈 Best R² Score: {grid_search.best_score_:.4f}")

final_model = Ridge(alpha=grid_search.best_params_['alpha'])

final_model.fit(X_train, y_train)

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("📊 Final Model Evaluation 📊")
print(f"✅ Final Train R² Score: {train_r2:.4f}")
print(f"✅ Final Test R² Score: {test_r2:.4f}")
print(f"✅ Final Train MAE: {train_mae:.4f}")
print(f"✅ Final Test MAE: {test_mae:.4f}")
print(f"✅ Final Train RMSE: {train_rmse:.4f}")
print(f"✅ Final Test RMSE: {test_rmse:.4f}")