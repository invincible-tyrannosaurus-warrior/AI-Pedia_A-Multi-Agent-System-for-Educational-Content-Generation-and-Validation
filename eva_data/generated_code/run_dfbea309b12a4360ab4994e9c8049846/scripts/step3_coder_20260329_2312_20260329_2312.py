import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import os

# Create the assets directory if it doesn't exist
assets_dir = '/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_dfbea309b12a4360ab4994e9c8049846/assets'
os.makedirs(assets_dir, exist_ok=True)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for easier handling
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]

print("Original dataset shape:", X.shape)
print("Features:", feature_names)

# Standardize the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Print explained variance information
print("\nExplained Variance Ratio:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.4f} ({cumulative_variance_ratio[i]:.4f} cumulative)")

# Plot 1: Explained variance by components
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Component')
plt.xticks(range(1, len(explained_variance_ratio) + 1))

# Plot 2: Cumulative explained variance
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'pca_explained_variance.png'))
plt.close()

# Plot 3: First two principal components colored by species
plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue']
for i, (color, target_name) in enumerate(zip(colors, target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                c=color, label=target_name, alpha=0.7)

plt.xlabel(f'First Principal Component (variance: {explained_variance_ratio[0]:.4f})')
plt.ylabel(f'Second Principal Component (variance: {explained_variance_ratio[1]:.4f})')
plt.title('PCA: First Two Principal Components')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(assets_dir, 'pca_components_scatter.png'))
plt.close()

# Plot 4: Feature contributions to first two components
# Get the components matrix
components = pca.components_

plt.figure(figsize=(12, 8))
x_pos = np.arange(len(feature_names))

# Plot for first component
plt.subplot(2, 1, 1)
plt.bar(x_pos, components[0], color='skyblue')
plt.xticks(x_pos, feature_names, rotation=45)
plt.ylabel('Contribution')
plt.title('First Principal Component Contributions')
plt.grid(axis='y', alpha=0.3)

# Plot for second component
plt.subplot(2, 1, 2)
plt.bar(x_pos, components[1], color='lightcoral')
plt.xticks(x_pos, feature_names, rotation=45)
plt.ylabel('Contribution')
plt.title('Second Principal Component Contributions')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'pca_feature_contributions.png'))
plt.close()

# Show how many components needed to explain 95% variance
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"\nNumber of components needed to explain 95% variance: {n_components_95}")

# Transform data to reduced dimensions (keeping 95% variance)
pca_reduced = PCA(n_components=0.95)
X_reduced = pca_reduced.fit_transform(X_scaled)

print(f"Reduced dataset shape: {X_reduced.shape}")
print(f"Original features: {len(feature_names)}")
print(f"Reduced features: {X_reduced.shape[1]}")

# Save the transformed data to CSV
reduced_df = pd.DataFrame(X_reduced, columns=[f'PC_{i+1}' for i in range(X_reduced.shape[1])])
reduced_df.to_csv(os.path.join(assets_dir, 'pca_reduced_data.csv'), index=False)

print("\nAnalysis complete. Files saved to:", assets_dir)