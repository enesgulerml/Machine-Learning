# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import missingno as msno

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Data Preparation
data = sns.load_dataset('penguins')
df = data.copy()

df.isnull().sum()
df.describe().T
df.nunique()
df.info()
df.head()

# Checking Outliers
def outlier_thresholds(dataframe, col_name, quantile1 = 0.01, quantile3 = 0.99):
    q1 = dataframe[col_name].quantile(quantile1)
    q3 = dataframe[col_name].quantile(quantile3)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound

def get_col_names(dataframe, cat_th=10, car_th=20):
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

num_cols

outlier_thresholds(df, "bill_length_mm")
outlier_thresholds(df, "bill_depth_mm")
outlier_thresholds(df, "flipper_length_mm")
outlier_thresholds(df, "body_mass_g")

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "body_mass_g")

# Checking Missing Values
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, na_name=True)

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

df["sex"].fillna(df["sex"].mode()[0], inplace=True)
for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)

missing_values_table(df)

# Visualization
sns.countplot(x="species", data=df, palette="viridis")
plt.title("Distribution of Penguin Species")
plt.show()

sns.boxplot(x="sex", y="body_mass_g",hue = "species", data=df, palette="coolwarm")
plt.title("Body Mass by Sex")
plt.show()

sns.scatterplot(x="bill_length_mm", y="bill_depth_mm", hue="species", data=df, palette="Set1")
plt.title("Bill Length vs. Bill Depth")
plt.show()

# Encoding

## Label Encoding
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df.head()
### Male == 1, Female == 0

## One-Hot Encoding
df = pd.get_dummies(df, columns=["island"], drop_first=True)
df.head()

# Modeling
## Random Forest
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("RandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))

### Hyperparameter Optimization
param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=23), param_grid_rf, cv=5, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

grid_search_rf.best_params_

y_pred_rf_best = grid_search_rf.best_estimator_.predict(X_test)
accuracy_score(y_test, y_pred_rf_best)

y_pred_proba_rf = rf_model.predict_proba(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba_rf, multi_class='ovr')
print("RandomForest ROC AUC Score:", roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_rf[:, 1], pos_label=1)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RandomForest ROC Curve')
plt.legend(loc='lower right')
plt.show()

print("RandomForest Classification Report:\n", classification_report(y_test, y_pred_rf))