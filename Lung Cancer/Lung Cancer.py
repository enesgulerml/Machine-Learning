import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 

from warnings import filterwarnings
filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float.format', lambda x: '{:.3f}'.format(x))


data = pd.read_csv("Projects/ML_Project/Lung Cancer/Lung Cancer Dataset.csv")
df = data.copy()

# Data Understanding

df.head()
df.info()
df.isnull().sum()
print(df.describe().T)
print(df.columns)

plt.figure(figsize=(10, 6))
sns.histplot(df['AGE'], kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='GENDER', data=df, palette='Set2')
plt.title('Gender Distribution')
plt.xlabel('Gender (0: Female, 1: Male)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='SMOKING', hue='PULMONARY_DISEASE', data=df, palette='Set1')
plt.title('Smoking vs Pulmonary Disease')
plt.xlabel('Smoking (0: No, 1: Yes)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='ALCOHOL_CONSUMPTION', hue='THROAT_DISCOMFORT', data=df, palette='muted')
plt.title('Alcohol Consumption vs Throat Discomfort')
plt.xlabel('Alcohol Consumption (0: No, 1: Yes)')
plt.ylabel('Count')
plt.show()

correlation_matrix = df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='PULMONARY_DISEASE', y='AGE', data=df, palette='Set2')
plt.title('Age Distribution by Pulmonary Disease Status')
plt.xlabel('Pulmonary Disease (NO: 0, YES: 1)')
plt.ylabel('Age')
plt.show()

# Model

X = df.drop(columns='PULMONARY_DISEASE',axis=1)
y = df['PULMONARY_DISEASE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Encoding

le=LabelEncoder()
y=le.fit_transform(y)

# RandomForestClassifier

rf_model=RandomForestClassifier(bootstrap=True)
rf_model.fit(X_train,y_train)

y_pred = rf_model.predict(X_test)

score=accuracy_score(y_test,y_pred)
print('Accuracy : ',score)

cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print('Confusion Matrix :\n',cm)
print('Classification Report :\n',cr)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.5, square=True)
plt.title('Random Forest Classifier Heat Map')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

## CV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

grid_search.fit(X, y)

print("Best Params: ", grid_search.best_params_)

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X)

accuracy = accuracy_score(y, y_pred)
print(f"Best Accuracy: {accuracy:.4f}")