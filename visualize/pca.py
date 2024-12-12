import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define the directory containing the latent vector files
latent_dir = "./latent_vectors"  # Update this path if necessary

# Load latent input and prediction files
latent_inputs = []
latent_predictions = []

for file in sorted(os.listdir(latent_dir)):
    if "latent_inputs" in file:
        latent_inputs.append(np.load(os.path.join(latent_dir, file)))
    elif "latent_predictions" in file:
        latent_predictions.append(np.load(os.path.join(latent_dir, file)))

# Concatenate all latent inputs and predictions into single arrays
latent_inputs = np.concatenate(latent_inputs, axis=0)  # Shape: (total_samples, latent_dim)
latent_predictions = np.concatenate(latent_predictions, axis=0)  # Shape: (total_samples, latent_dim)

# Reshape latent_inputs and latent_predictions to 2D
latent_inputs = latent_inputs.reshape(-1, latent_inputs.shape[-1])  # Flatten batch/time dimensions
latent_predictions = latent_predictions.reshape(-1, latent_predictions.shape[-1])  # Flatten batch/time dimensions

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
latent_inputs_pca = pca.fit_transform(latent_inputs)
latent_predictions_pca = pca.transform(latent_predictions)

# Plot the PCA results
plt.figure(figsize=(10, 7))

plt.scatter(
    latent_inputs_pca[:, 0], latent_inputs_pca[:, 1],
    alpha=0.6, label='Latent Inputs', c='blue', edgecolors='k'
)

plt.scatter(
    latent_predictions_pca[:, 0], latent_predictions_pca[:, 1],
    alpha=0.6, label='Latent Predictions', c='red', edgecolors='k'
)
plt.title("PCA of Latent Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()
