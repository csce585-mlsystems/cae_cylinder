import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Define the directory containing the latent vector files
latent_dir = "./latent_vectors"  # Update this path as necessary

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

# Flatten batch and time dimensions (if necessary)
latent_inputs = latent_inputs.reshape(-1, latent_inputs.shape[-1])  # Ensure 2D array
latent_predictions = latent_predictions.reshape(-1, latent_predictions.shape[-1])  # Ensure 2D array

# Combine inputs and predictions for visualization
combined_latents = np.concatenate([latent_inputs, latent_predictions], axis=0)
labels = np.array([0] * latent_inputs.shape[0] + [1] * latent_predictions.shape[0])  # 0 for inputs, 1 for predictions

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
latent_tsne = tsne.fit_transform(combined_latents)

# Split the t-SNE results for plotting
latent_inputs_tsne = latent_tsne[labels == 0]
latent_predictions_tsne = latent_tsne[labels == 1]

# Plot the t-SNE results
plt.figure(figsize=(10, 7))
plt.scatter(
    latent_inputs_tsne[:, 0], latent_inputs_tsne[:, 1],
    alpha=0.6, label='Latent Inputs', c='blue', edgecolors='k'
)

plt.scatter(
    latent_predictions_tsne[:, 0], latent_predictions_tsne[:, 1],
    alpha=0.6, label='Latent Predictions', c='red', edgecolors='k'
)
plt.title("t-SNE of Latent Space")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.grid(True)
plt.show()
