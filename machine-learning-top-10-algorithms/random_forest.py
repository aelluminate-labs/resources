import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Generate a synthetic dataset
X, y = make_classification(n_samples=2000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# Predict the test set results
y_pred = rf.predict(X_test)

# Visualize the results
plt.figure(figsize=(10, 7))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
plt.title('Random Forest Classification Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Predicted Class')
plt.show()