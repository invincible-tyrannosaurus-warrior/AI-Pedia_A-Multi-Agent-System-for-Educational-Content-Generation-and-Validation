import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import os

# Set the output directory
output_dir = '/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_39f020bb65e9481cb311c11f0d59c8f4/assets'
os.makedirs(output_dir, exist_ok=True)

# Generate sample data using make_blobs for better visualization
# This creates 400 samples with 2 features and 4 centers (true clusters)
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)

# Create a DataFrame for easier handling
df = pd.DataFrame(X, columns=['feature1', 'feature2'])

# Display first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Standardize the features to have zero mean and unit variance
# This is important for K-means clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering with k=4 (since we know there are 4 true clusters)
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_pred = kmeans.fit_predict(X_scaled)

# Get the cluster centers in the original scale
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)

# Plot the results
plt.figure(figsize=(12, 5))

# Plot 1: Original data with true labels
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
plt.title('Original Data with True Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot 2: K-means clustering result
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
plt.scatter(centers_original[:, 0], centers_original[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'kmeans_clustering_results.png'))
plt.close()

# Calculate and display clustering metrics
from sklearn.metrics import silhouette_score

# Compute silhouette score to evaluate clustering quality
silhouette_avg = silhouette_score(X_scaled, y_pred)
print(f"Average silhouette score: {silhouette_avg:.3f}")

# Find optimal number of clusters using the elbow method
# We'll test different values of k from 1 to 10
k_range = range(1, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    
    # Only compute silhouette score for k > 1
    if k > 1:
        score = silhouette_score(X_scaled, kmeans_temp.labels_)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

# Plot the elbow curve
plt.figure(figsize=(10, 5))

# Plot 1: Elbow method
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

# Plot 2: Silhouette scores
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores[1:], 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'elbow_method_and_silhouette.png'))
plt.close()

# Print information about the clustering process
print("\nClustering Analysis Summary:")
print("-" * 30)
print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Optimal number of clusters based on silhouette score: {np.argmax(silhouette_scores) + 2}")
print(f"Silhouette score for optimal k: {max(silhouette_scores):.3f}")

# Save the clustered data to CSV
df_clustered = df.copy()
df_clustered['cluster'] = y_pred
df_clustered.to_csv(os.path.join(output_dir, 'clustered_data.csv'), index=False)
print(f"\nClustered data saved to {os.path.join(output_dir, 'clustered_data.csv')}")

# Display cluster statistics
print("\nCluster Statistics:")
print("-" * 20)
for i in range(4):
    cluster_points = df_clustered[df_clustered['cluster'] == i]
    print(f"Cluster {i}: {len(cluster_points)} points")
    print(f"  Mean feature1: {cluster_points['feature1'].mean():.2f}")
    print(f"  Mean feature2: {cluster_points['feature2'].mean():.2f}")
    print()