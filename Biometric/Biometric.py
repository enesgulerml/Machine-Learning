# Importing Necessary Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix

sns.set_theme(style="darkgrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df = pd.read_csv("Projects/ML/Biometric/column_2C_weka.csv")

df.head()
df.shape
df.info()
df.columns
df.isnull().sum()
df.nunique()
df.describe().T

# Let's see some graphics

color_map = {'Abnormal': 'purple', 'Normal': 'orange'}
color_list = [color_map[i] for i in df['class']]

scatter_matrix(df.iloc[:, :-1],
               c=color_list,
               figsize=[12, 12],
               diagonal='kde',
               alpha=0.7,
               s=100,
               marker='o',
               edgecolors="gray")
plt.show()
plt.xticks(rotation=90)

dark_colors = ["#4B0082", "#8B0000"]

plt.figure(figsize=(8, 6))
sns.countplot(x="class", data=df, palette=dark_colors)
plt.title("Class Distribution", fontsize=14, fontweight="bold", color="white")
plt.xlabel("Class", fontsize=12, color="white")
plt.ylabel("Count", fontsize=12, color="white")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

plt.figure(figsize=(6, 6))
df['class'].value_counts().plot.pie(autopct='%1.1f%%', colors=['red', 'blue'], startangle=90)
plt.title("Class Distribution", fontsize=14, fontweight="bold", color="white")
plt.ylabel("")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Model (KNN)

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(df.drop('class', axis=1), df['class'])
print('Prediction:', knn.predict(df.drop('class', axis=1)))

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(df.drop('class', axis=1), df['class'], test_size=0.2, random_state=23)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(f'Accuracy with KNN (K=3): {knn.score(X_test, y_test):.2f}')
