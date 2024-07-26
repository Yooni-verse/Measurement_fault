import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Dummy data
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.title('HDBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label')
plt.show()
