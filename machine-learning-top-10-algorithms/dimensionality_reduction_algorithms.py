import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS, TSNE
 
# Generate a synthetic dataset
X, y = make_classification(n_samples=2000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
 
# Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
 
# Reduce dimensionality for visualization using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
 
# Reduce dimensionality for visualization using Isomap
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X)
 
# Reduce dimensionality for visualization using MDS
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)
 
# Visualize the results
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
 
# PCA
axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
axs[0, 0].set_title('PCA')
axs[0, 0].set_xlabel('Principal Component 1')
axs[0, 0].set_ylabel('Principal Component 2')
 
# t-SNE
axs[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
axs[0, 1].set_title('t-SNE')
axs[0, 1].set_xlabel('Component 1')
axs[0, 1].set_ylabel('Component 2')
 
# Isomap
axs[1, 0].scatter(X_isomap[:, 0], X_isomap[:, 1], c=y, cmap='viridis', alpha=0.7)
axs[1, 0].set_title('Isomap')
axs[1, 0].set_xlabel('Component 1')
axs[1, 0].set_ylabel('Component 2')
 
# MDS
axs[1, 1].scatter(X_mds[:, 0], X_mds[:, 1], c=y, cmap='viridis', alpha=0.7)
axs[1, 1].set_title('MDS')
axs[1, 1].set_xlabel('Component 1')
axs[1, 1].set_ylabel('Component 2')
 
plt.tight_layout()
plt.show()