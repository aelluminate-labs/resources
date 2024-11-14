import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import plot_tree
 
# Generate a synthetic dataset
X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42)
 
# Train a Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=3, random_state=42)
gb_clf.fit(X, y)
 
# Train a Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=3, random_state=42)
rf_clf.fit(X, y)
 
# Train an AdaBoost Classifier
ab_clf = AdaBoostClassifier(n_estimators=3, random_state=42)
ab_clf.fit(X, y)
 
# Plot the individual trees for Gradient Boosting
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
for i, ax in enumerate(axes):
    plot_tree(gb_clf.estimators_[i, 0], ax=ax, filled=True, feature_names=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
    ax.set_title(f'Gradient Boosting Tree {i+1}')
 
plt.tight_layout()
plt.show()
 
# Plot the individual trees for Random Forest
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
for i, ax in enumerate(axes):
    plot_tree(rf_clf.estimators_[i], ax=ax, filled=True, feature_names=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
    ax.set_title(f'Random Forest Tree {i+1}')
 
plt.tight_layout()
plt.show()
 
# Plot the individual trees for AdaBoost
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
for i, ax in enumerate(axes):
    plot_tree(ab_clf.estimators_[i], ax=ax, filled=True, feature_names=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
    ax.set_title(f'AdaBoost Tree {i+1}')
 
plt.tight_layout()
plt.show()