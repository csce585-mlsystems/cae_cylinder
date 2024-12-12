import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from process import load_data
from cae import CVAutoencoder
from train import encoder_path, decoder_path, lstm_path


startTime = 100.067
dt = 0.077                          # time interval between snapshots
RE = 160
file_path = "data/40-data-100.375"  # file to test CAE reconstruction

sequence_path = [                   # specify what files to input to test LSTM prediction (can be any #)
    "validate/180-data-100.067",
    "validate/180-data-100.144",
    "validate/180-data-100.221",
    "validate/180-data-100.298",
    "validate/180-data-100.375",
    "validate/180-data-100.452",
    "validate/180-data-100.529",
    "validate/180-data-100.606",
    "validate/180-data-100.683",
    "validate/180-data-100.760",
    "validate/180-data-100.837",
    "validate/180-data-100.914",
    "validate/180-data-100.991",
    "validate/180-data-101.068",
    "validate/180-data-101.145",
    "validate/180-data-101.222",
    "validate/180-data-101.299",
    "validate/180-data-101.376",
    "validate/180-data-101.453",
    "validate/180-data-101.530",
    "validate/180-data-101.607"
]


re_mean = 120    # precalculated based on even number of files for Re={40,80,120,160,200}
re_std = 56.5685
re_norm = (RE - re_mean) / re_std


def decode_and_plot_comparison(file_path, encoder_path, decoder_path, device="cuda"):

    # Load the pretrained autoencoder
    device = torch.device(device)
    ae = CVAutoencoder().to(device)
    ae.encoder.load_state_dict(torch.load(encoder_path))
    ae.decoder.load_state_dict(torch.load(decoder_path))
    ae.encoder.eval()
    ae.decoder.eval()

    # Load the snapshot
    snapshot = load_data(file_path).unsqueeze(0).to(device)  # Add batch dimension, shape: [1, 3, height, width]

    # Encode and decode the snapshot
    with torch.no_grad():
        latent_vector = ae.encoder(snapshot)  # Shape: [1, latent_dim]
        decoded_snapshot = ae.decoder(latent_vector)  # Shape: [1, 3, height, width]

    # Remove batch dimension for plotting
    input_snapshot = snapshot.squeeze(0).cpu().numpy()  # Shape: [3, height, width]
    output_snapshot = decoded_snapshot.squeeze(0).cpu().numpy()  # Shape: [3, height, width]

    # Compute the absolute error
    error_snapshot = np.abs(input_snapshot - output_snapshot)

    components = ['u', 'v', 'p']
    for i, component in enumerate(components):
        plt.figure(figsize=(15, 5))

        # Input
        plt.subplot(1, 3, 1)
        plt.imshow(input_snapshot[i], cmap="viridis")
        plt.colorbar()
        plt.title(f"Input: {component}")

        # Output
        plt.subplot(1, 3, 2)
        plt.imshow(output_snapshot[i], cmap="viridis")
        plt.colorbar()
        plt.title(f"Decoded Output: {component}")

        # Error
        plt.subplot(1, 3, 3)
        plt.imshow(error_snapshot[i], cmap="inferno")
        plt.colorbar()
        plt.title(f"Absolute Error: {component}")

        plt.tight_layout()
        plt.show()
        plt.save(f"cae_output_{component}")

def recursive_validation_with_plots(
    encoder_path,
    decoder_path,
    lstm_path,
    file_paths,
    re_value,
    num_predictions,
    ground_truth_dir,
    output_dir="predicted_images",
    device="cuda",
    plot_after=5,  # Number of recursive predictions to wait before plotting
    start_time=startTime
):

    # Load pretrained models
    device = torch.device(device)
    ae = CVAutoencoder().to(device)
    ae.encoder.load_state_dict(torch.load(encoder_path))
    ae.decoder.load_state_dict(torch.load(decoder_path))
    ae.encoder.eval()
    ae.decoder.eval()

    lstm = ae.lstm
    lstm.load_state_dict(torch.load(lstm_path))
    lstm.eval()

    # Prepare initial inputs (first snapshots in sequence)
    initial_snapshots = torch.stack([load_data(fp) for fp in file_paths]).to(device)  # Shape: [5, 3, height, width]

    # Encode the initial snapshots
    initial_latents = torch.stack(
        [ae.encoder(snapshot.unsqueeze(0)).squeeze(0) for snapshot in initial_snapshots]
    )  # Shape: [5, latent_dim]

    # Reynolds number tensor
    re_value_tensor = torch.tensor(
        [[[re_value]] * initial_latents.size(0)], device=device, dtype=torch.float32
    )  # Shape: [1, seq_length, 1]

    for step in range(num_predictions):
        # Prepare input for LSTM
        input_sequence = initial_latents.unsqueeze(0)              # Shape: [1, seq_length, latent_dim]

        # Predict the next latent vector
        with torch.no_grad():
            next_latent = lstm(input_sequence, re_value_tensor)    # Shape: [1, 1, latent_dim]
            predicted_last_timestep = next_latent[:, -1, :]        # Shape: [batch_size, latent_dim]
            predicted_field = ae.decoder(predicted_last_timestep)  # Decode to high-dimensional flow field

        # Save the predicted field
        predicted_file = os.path.join(output_dir, f"predicted_{step + 1}.npz")
        predicted_field_np = predicted_field.detach().cpu().numpy()
        np.savez(
            predicted_file,
            u_x=predicted_field_np[0, 0],
            u_y=predicted_field_np[0, 1],
            p=predicted_field_np[0, 2],
        )

        # Load the ground truth field
        ground_truth_file = os.path.join(
            ground_truth_dir, f"{re_value}-data-{startTime + dt * (step + 1):.3f}"
        )
        ground_truth_field = load_data(ground_truth_file).detach().cpu().numpy()

        # Calculate and log MSE
        mse = np.mean((predicted_field_np - ground_truth_field) ** 2)
        print(f"Step {step + 1}: MSE with ground truth: {mse}")

        # Plot after the specified number of steps
        if (step + 1) % plot_after == 0 or (step + 1) == num_predictions:
            components = ["u", "v", "p"]
            for i, component in enumerate(components):
                plt.figure(figsize=(15, 5))

                # Ground Truth
                plt.subplot(1, 3, 1)
                plt.imshow(ground_truth_field[i], cmap="viridis")
                plt.colorbar()
                plt.title(f"Ground Truth: {component} (Step {step + 1})")

                # Predicted Field
                plt.subplot(1, 3, 2)
                plt.imshow(predicted_field_np[0, i], cmap="viridis")
                plt.colorbar()
                plt.title(f"Predicted: {component} (Step {step + 1})")

                # Error
                plt.subplot(1, 3, 3)
                error_snapshot = np.abs(ground_truth_field[i] - predicted_field_np[0, i])
                plt.imshow(error_snapshot, cmap="inferno")
                plt.colorbar()
                plt.title(f"Absolute Error: {component} (Step {step + 1})")

                plt.tight_layout()
                plt.show()

        # Update the sequence for the next prediction
        initial_latents = torch.cat((initial_latents[1:], predicted_last_timestep), dim=0)

###### Uncomment to test CAE given a file (file_path) ######
# decode_and_plot_comparison(file_path, encoder_path, decoder_path, device="cuda")

##### Uncomment to test CAE w/LSTM given a sequence (sequence_path) ######
'''
recursive_validation_with_plots(
    encoder_path,
    decoder_path,
    lstm_path,
    file_paths=sequence_path,  # Initial snapshots
    re_value=re_norm,  # Normalized Reynolds number
    num_predictions=100,  # Number of recursive predictions
    ground_truth_dir="data/",
    output_dir="predicted_snapshots",
    device="cuda",
    plot_after=10,  # Generate plots after every 10 predictions
    start_time=startTime
)
'''