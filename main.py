import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

# Suppress warnings (optional)
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv('winequality-red.csv', sep=',', header=0)


# Convert all columns to numeric (in case of bad values)
data = data.apply(pd.to_numeric, errors='coerce')

# Display basic info
print(data.info())
print(data.describe())
print("Missing values in each column:\n", data.isnull().sum())

# Split into features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
print("\n📊 Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
print("\n🌳 Decision Tree Accuracy:", accuracy_score(y_test, dt_predictions))
print(classification_report(y_test, dt_predictions))

# KMeans Clustering (on scaled data)
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)

# Plot clustering
sns.scatterplot(data=data, x='fixed acidity', y='citric acid', hue='cluster', palette='viridis')
plt.title('K-Means Clustering of Wine Quality')
plt.savefig("kmeans_plot.png")  # Save the plot as an image (better for Docker/CI/CD)

# Save models
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(dt_model, 'dt_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Optional: Save the scaler too

print("\n✅ Models and scaler saved successfully.")
