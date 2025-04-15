import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

data = pd.read_csv('winequality-red.csv', sep=';', header=0)

data = data['fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality'].str.split(',', expand=True)
data.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

print(data.info())
print(data.describe())
print("Missing values in each column:\n", data.isnull().sum())

X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_train_score = lr_model.score(X_train, y_train)
lr_test_score = lr_model.score(X_test, y_test)
print(f"Logistic Regression Train Accuracy: {lr_train_score:.2f}")
print(f"Logistic Regression Test Accuracy: {lr_test_score:.2f}")
print("Logistic Regression Classification Report:\n", classification_report(y_test, lr_model.predict(X_test)))
joblib.dump(lr_model, 'lr_model.pkl')

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_train_score = dt_model.score(X_train, y_train)
dt_test_score = dt_model.score(X_test, y_test)
print(f"Decision Tree Train Accuracy: {dt_train_score:.2f}")
print(f"Decision Tree Test Accuracy: {dt_test_score:.2f}")
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_model.predict(X_test)))
joblib.dump(dt_model, 'dt_model.pkl')

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X['alcohol'], y=y, hue=labels, palette='viridis')
plt.title('K-Means Clustering of Wine Quality')
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.show()