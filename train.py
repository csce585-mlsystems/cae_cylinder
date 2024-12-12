import glob
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Subset, DataLoader
from process import  PretrainFlowDataset, SequenceDataset, load_data
from cae import CVAutoencoder, BatchSize, new_latent_dimension

Epochs = 1500
sequence_length = 21
num_predictions = 15

encoder_path = "weights/pretrained_encoder.pth"
decoder_path = "weights/pretrained_decoder.pth"
lstm_path = "weights/trained_lstm.pth"

file_paths = sorted(glob.glob("data/*-data-*"))

# List of initial snapshots for LSTM validation
init_file_paths = [
    "validate/20-data-46.700",
    "validate/20-data-46.777",
    "validate/20-data-46.854",
    "validate/20-data-46.931",
    "validate/20-data-47.008"
]

init_file_paths = [
    "validate/180-data-46.700",
    "validate/180-data-46.777",
    "validate/180-data-46.854",
    "validate/180-data-46.931",
    "validate/180-data-47.008"
]

init_file_paths = [
    "data/120-data-100.067",
    "data/120-data-100.144",
    "data/120-data-100.221",
    "data/120-data-100.298",
    "data/120-data-100.375"
]

file_path = "data/40-data-100.375"

sequence_path = [
    "data/120-data-100.067",
    "data/120-data-100.144",
    "data/120-data-100.221",
    "data/120-data-100.298",
    "data/120-data-100.375",
    "data/120-data-100.452",
    "data/120-data-100.529",
    "data/120-data-100.606",
    "data/120-data-100.683",
    "data/120-data-100.760",
    "data/120-data-100.837",
    "data/120-data-100.914",
    "data/120-data-100.991",
    "data/120-data-101.068",
    "data/120-data-101.145",
    "data/120-data-101.222",
    "data/120-data-101.299",
    "data/120-data-101.376",
    "data/120-data-101.453",
    "data/120-data-101.530",
    "data/120-data-101.607"
]
"""
sequence_path = [
    "data/120-data-100.067",
    "data/120-data-100.144",
    "data/120-data-100.221",
    "data/120-data-100.298",
    "data/120-data-100.375",
    "data/120-data-100.452",
    "data/120-data-100.529",
    "data/120-data-100.606",
    "data/120-data-100.683",
    "data/120-data-100.760",
    "data/120-data-100.837",
    "data/120-data-100.914",
    "data/120-data-100.991",
    "data/120-data-101.068",
    "data/120-data-101.145",
    "data/120-data-101.222",
    "data/120-data-101.299",
    "data/120-data-101.376",
    "data/120-data-101.453",
    "data/120-data-101.530",
    "data/120-data-101.607"
]
"""
# '''
sequence_path = [
    "validate/20-data-46.700",
    "validate/20-data-46.777",
    "validate/20-data-46.854",
    "validate/20-data-46.931",
    "validate/20-data-47.008"
]
# '''
'''
sequence_path = [
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
'''
"""
sequence_path = [
    "validate/180-data-46.700",
    "validate/180-data-46.777",
    "validate/180-data-46.854",
    "validate/180-data-46.931",
    "validate/180-data-47.008"
]
"""

# print(file_paths)

# re_numbers = [Re] * len(file_paths)
# re_numbers = [int(file.split('-')[0]) for file in file_paths]
re_numbers = [int(file.split('/')[1].split('-')[0]) for file in file_paths]
mean_re = np.mean(re_numbers)
std_re = np.std(re_numbers)

re_val = 120
re_norm = (re_val - mean_re) / std_re

print(re_norm)

# print (re_numbers)

re_normalize = [(re - mean_re) / std_re for re in re_numbers]
# print (re_normalize)



# dataset = FlowDataset(file_paths, re_numbers, sequence_length)
# dataset = PretrainFlowDataset(file_paths, re_normalize)
print("sequence start")
# dataset = SequenceDataset(file_paths, re_normalize, sequence_length)

print("sequence end")
# Normalzie Reynold #s
# mean_re = np.mean(dataset.re_labels)
# std_re = np.std(dataset.re_labels)
# dataset.re_labels= [(re - mean_re) / std_re for re in dataset.re_labels]

# print(dataset.sequences.shape)

# dataloader = DataLoader(dataset, batch_size=BatchSize, shuffle=True)  # Adjust batch size as needed

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
ae = CVAutoencoder().to(device)

def denormalize(image, mean, std):
    return image * std + mean


def four_fold_cross_validation(model_class, dataset, num_epochs=100, batch_size=32, learning_rate=1e-4, device="cuda"):
    """
    Perform 4-fold cross-validation on the given dataset.

    Parameters:
        model_class: The class of the model to be trained (e.g., CVAutoencoder).
        dataset: The full dataset to be split into folds.
        num_epochs: Number of epochs for training in each fold.
        batch_size: Batch size for DataLoader.
        learning_rate: Learning rate for the optimizer.
        device: The device to use ("cuda" or "cpu").

    Returns:
        float: The average validation loss across all folds.
    """
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    fold_results = []

    loss_history = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        # print(f"Starting Fold {fold + 1}")

        # Split dataset into training and validation subsets
        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        # Initialize a new model for each fold
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            i_batch = 0
            for batch in train_loader:
                i_batch += 1
                inputs, re_values = batch  # Unpack flow fields and Reynolds numbers
                inputs = inputs.to(device)
                re_values = re_values.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                latent = model.encode(inputs)  # Encode
                outputs = model.decode(latent)  # Decode
                loss = criterion(outputs, inputs)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                """
                if epoch == num_epochs - 1:
                  model.eval()
                  save_flow_fields(inputs=inputs, outputs=outputs, timestep=i_batch, fold=+1, val=0, re=re_values)
                  model.train()
                """

            average_loss = epoch_loss / len(train_loader)
            loss_history.append(average_loss)  # Save the average loss        
            
            print(f"{fold + 1} {epoch + 1} {epoch_loss / len(train_loader):.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, re_values = batch  # Unpack flow fields and Reynolds numbers
                inputs = inputs.to(device)
                re_values = re_values.to(device)

                latent = model.encode(inputs)  # Encode
                outputs = model.decode(latent)  # Decode
                val_loss += criterion(outputs, inputs).item()
                ## save_flow_fields(inputs=inputs, outputs=outputs, timestep=i_batch, fold=fold+1, val=1, re=re_values)
        
        val_loss /= len(val_loader)
        print(f"Validation Loss for Fold {fold + 1}: {val_loss:.4f}")
        fold_results.append(val_loss)

    with open("CAE_loss_history.csv", "w", newline="") as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(["Epoch", "Loss"])
      for epoch, loss in enumerate(loss_history, 1):
        writer.writerow([epoch, loss])

    with open("fold_loss_history.csv", "w", newline="") as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(["Epoch", "Loss"])
      for epoch, loss in enumerate(fold_results, 1):
        writer.writerow([epoch, loss])
  
    # Calculate and return the average validation loss across all folds
    average_loss = np.mean(fold_results)
    print(f"Average Validation Loss Across Folds: {average_loss:.4f}")
    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    print(f"Saved encoder and decoder for Fold {fold + 1}.")
    return average_loss

def train_lstm(lstm_model, dataset, encoder, device, latent_dim, batch_size=16, epochs=10, learning_rate=1e-3):
    """
    Train an LSTM model to predict the next latent vector in a sequence.

    Args:
        lstm_model (nn.Module): LSTM model to train.
        dataset (Dataset): Dataset providing sequences of flow fields and Reynolds numbers.
        encoder (nn.Module): Pretrained encoder to encode flow field snapshots into latent vectors.
        device (str): Device to use ('cuda' or 'cpu').
        latent_dim (int): Dimension of the latent space.
        batch_size (int, optional): Batch size for training. Defaults to 16.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.

    Returns:
        lstm_model (nn.Module): Trained LSTM model.
    """

    latent_save_dir = "latent_vectors"
    os.makedirs(latent_save_dir, exist_ok=True)

    train_indices, val_indices = train_test_split(
        list(range(len(dataset0))), test_size=0.2, random_state=42
    )
    train_dataset = Subset(dataset0, train_indices)
    val_dataset = Subset(dataset0, val_indices)

    # DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training dataset: {train_loader.sequences.shape} | Validation dataset: {val_loader.sequences.shape}")

    # Load pretrained weights for encoder and decoder
    ae.encoder.load_state_dict(torch.load(encoder_path))
    ae.decoder.load_state_dict(torch.load(decoder_path))

    # DataLoader
    precompute_latent_vectors(dataset, encoder, device, latent_dim, save_path="precomputed_latents.pt")
    precomputed_latents, precomputed_re_values = load_precomputed_latents("precomputed_latents.pt")
    dataset0 = list(zip(precomputed_latents, precomputed_re_values))

    # Prepare optimizer and loss function
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #            optimizer, mode='min', factor=0.9, patience=30)

    loss_history = []  # List to store the loss values for each epoch
    val_loss_history = []  # List to store the validation loss valuesloss_history = []  # List to store the loss values for each epoch

    # Training Loop
    for epoch in range(epochs):
        lstm_model.train()
        total_loss = 0

        i_sequence = 0
        for sequences, re_values in train_loader:
            i_sequence += 1

            sequences = sequences.to(device)  # Shape: [batch_size, sequence_length, latent_dim]
            re_values = re_values.to(device)

            
            # Expand re_values to match [batch_size, sequence_length, 1]
            current_batch_size, current_sequence_length, _= sequences.size()
            re_values = re_values.unsqueeze(1).expand(current_batch_size, current_sequence_length - 1, 1)  # Expand to [batch_size, sequence_length - 1, 1]
            
            optimizer.zero_grad()
           
            # Input: first 20 timesteps; Target: 21st timestep
            input_sequence = sequences[:, :-1, :]  # All but the last
            target_sequence = sequences[:, -1, :].unsqueeze(1)  # Only the last

            # Forward Pass
            predicted_sequence = lstm_model(input_sequence, re_values)


            # Select the last timestep from the predicted sequence
            predicted_last_timestep = predicted_sequence[:, -1, :]  # Shape: [batch_size, latent_dim]

            # Compute the loss using the last timestep
            
            loss = criterion(predicted_last_timestep, target_sequence.squeeze(1))  # Ensure target_sequence is the same shape
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if epoch==Epochs-1 & i_sequence < BatchSize:
                latent_input_file = os.path.join(latent_save_dir, f"latent_inputs_epoch{epoch}.npy")
                latent_predicted_file = os.path.join(latent_save_dir, f"latent_predictions_epoch{epoch}.npy")
                np.save(latent_input_file, input_sequence.detach().cpu().numpy())
                np.save(latent_predicted_file, predicted_sequence.detach().cpu().numpy())
            
        average_loss = total_loss / len(train_loader)
        loss_history.append(average_loss)  # Save the average loss

        # Validation Phase
        lstm_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for sequences, re_values in val_loader:
                sequences = sequences.to(device)
                re_values = re_values.to(device)

                current_batch_size, current_sequence_length, _= sequences.size()
                re_values = re_values.unsqueeze(1).expand(current_batch_size, current_sequence_length - 1, 1)  # Expand to [batch_size, sequence_length - 1, 1]

                input_sequence = sequences[:, :-1, :]
                target_sequence = sequences[:, -1, :].unsqueeze(1)

                predicted_sequence = lstm_model(input_sequence, re_values)
                predicted_last_timestep = predicted_sequence[:, -1, :]

                val_loss = criterion(predicted_last_timestep, target_sequence.squeeze(1))
                total_val_loss += val_loss.item()

        average_val_loss = total_loss / len(val_loader)
        val_loss_history.append(average_val_loss)

        # scheduler.step(average_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {average_loss:.8f}, Val Loss: {average_val_loss:.8f}")
    
    # Save loss histories to CSV
    with open("loss_history.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])
        for epoch, (train_loss, val_loss) in enumerate(zip(loss_history, val_loss_history), 1):
            writer.writerow([epoch, train_loss, val_loss])

    torch.save(lstm_model.state_dict(), lstm_path)
    print(f"Trained lstm saved to {lstm_path}")

    return lstm_model

def precompute_latent_vectors(dataset, encoder, device, latent_dim, save_path="precomputed_latents.pt"):

    encoder.eval()
    all_latents = []
    all_re_values = []
    
    with torch.no_grad():
        for sequences, re_values in DataLoader(dataset, batch_size=1):
            sequences = sequences.to(device)  # Shape: [1, sequence_length, 3, height, width]
            re_values = re_values.to(device)

            # Encode each sequence
            latent_vectors = torch.stack([
                encoder(frame.unsqueeze(0)).squeeze(0) for frame in sequences[0]
            ])  # Shape: [sequence_length, latent_dim]

            all_latents.append(latent_vectors.cpu())
            all_re_values.append(re_values.cpu())

    # Save latent vectors and Re values
    torch.save({"latents": all_latents, "re_values": all_re_values}, save_path)
    print(f"Precomputed latent vectors saved to {save_path}.")

def load_precomputed_latents(save_path):

    data = torch.load(save_path)
    return data["latents"], data["re_values"]


def preprocess_sequence(sequence, encoder, device):

    latent_vectors = []
    for frame in sequence:  # Process each timestep in the sequence
        frame = frame.unsqueeze(0).to(device)  # Add batch dimension
        latent = encoder(frame)
        latent_vectors.append(latent.squeeze(0))  # Remove batch dimension
    return torch.stack(latent_vectors)  # Shape: [sequence_length, latent_dim]

def recursive_validation(encoder_path, decoder_path, lstm_path, file_paths, re_value, num_predictions, ground_truth_dir, output_dir="predicted_snapshots", device="cuda"):

    # Load pretrained models
    device = torch.device(device)
    ae = CVAutoencoder().to(device)
    ae.encoder.load_state_dict(torch.load(encoder_path))
    ae.decoder.load_state_dict(torch.load(decoder_path))
    
    lstm = ae.lstm
    lstm.load_state_dict(torch.load(lstm_path))
    lstm.eval()

    # Prepare initial inputs 
    initial_snapshots = torch.stack([load_data(fp) for fp in file_paths]).to(device)

    # Encode the initial snapshots
    initial_latents = torch.stack([ae.encoder(snapshot.unsqueeze(0)).squeeze(0) for snapshot in initial_snapshots])

    re_value_tensor = torch.tensor(
      [[[re_value]] * initial_latents.size(0)],
      device=device,
      dtype=torch.float32
      )  # Shape: [1, 5, 1]
    print(re_value_tensor.shape)

    for step in range(num_predictions):
        # Prepare input for LSTM
        input_sequence = initial_latents.unsqueeze(0)

        # Predict the next latent vector
        next_latent = lstm(input_sequence, re_value_tensor) 
        predicted_last_timestep = next_latent[:, -1, :] 
        # Decode to high-dimensional flow field
        predicted_field = ae.decoder(next_latent.squeeze(0)) 
  
        # Save the predicted field
        predicted_file = os.path.join(output_dir, f"{re_val}-predicted_{step + 6}.npz")  # Start from 6th snapshot
        # Move predicted_field to CPU and convert to NumPy
        predicted_field_np = predicted_field.detach().cpu().numpy()
        np.savez(predicted_file, u_x=predicted_field_np[0, 0], u_y=predicted_field_np[0, 1], p=predicted_field_np[0, 2])


        # Load the ground truth field
        ground_truth_file = os.path.join(ground_truth_dir, f"{re_val}-data-{100.067 + 0.077 * (step + 1):.3f}")
        ground_truth_field = load_data(ground_truth_file).detach().cpu().numpy()

        # Calculate difference (e.g., MSE)
        # Ensure both are NumPy arrays

        mse = np.mean((predicted_field_np - ground_truth_field) ** 2)

        print(f"Step {step + 6}: MSE with ground truth (t={100.067 + 0.077 * (step + 1):.3f}): {mse}")
        
        # Update the sequence by removing the oldest latent and adding the new one
        initial_latents = torch.cat((initial_latents[1:], predicted_last_timestep), dim=0)

    # return predicted_fields

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

    # Plot the comparison for each component (u_x, u_y, p)
    # components = ['u_x', 'u_y', 'p']
    components = ['u_x']
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

def plot_lstm_prediction(sequence_path, encoder_path, decoder_path, lstm_path, device="cuda"):

    # Load the pretrained autoencoder and LSTM
    device = torch.device(device)
    ae = CVAutoencoder().to(device)

    ae.encoder.load_state_dict(torch.load(encoder_path))
    ae.decoder.load_state_dict(torch.load(decoder_path))
    ae.lstm.load_state_dict(torch.load(lstm_path))

    ae.encoder.eval()
    ae.decoder.eval()
    ae.lstm.eval()

    # Encode the sequence into latent space
    latent_sequence = []
    for file in sequence_path[:-1]:  # Use first 20 snapshots as input
        snapshot = load_data(file).unsqueeze(0).to(device)  # Shape: [1, 3, height, width]
        with torch.no_grad():
            latent_vector = ae.encoder(snapshot)  # Encode to latent space
        latent_sequence.append(latent_vector)

    latent_sequence = torch.cat(latent_sequence, dim=0).unsqueeze(0)  # Shape: [1, seq_len, latent_dim]

    _ , seq_length, _ = latent_sequence.size()

    re_tensor = torch.full((1, seq_length , 1), re_val, device=device)  # Shape: [1, seq_length, 1]


    # Predict the 21st latent vector with the LSTM
    with torch.no_grad():
        lstm_predicted_latent = ae.lstm(latent_sequence, re_tensor)  # Shape: [1, latent_dim]
    

    predicted_last_timestep = lstm_predicted_latent[:, -1, :]  # Shape: [batch_size, latent_dim]
    # Decode the predicted latent vector
    with torch.no_grad():
        predicted_snapshot = ae.decoder(predicted_last_timestep)  # Shape: [1, 3, height, width]

    # Decode the ground truth 21st snapshot
    ground_truth_snapshot = load_data(sequence_path[-1]).unsqueeze(0).to(device)  # Ground truth snapshot
    with torch.no_grad():
        true_snapshot = ae.decoder(ae.encoder(ground_truth_snapshot))  # Ensure ground truth is decoded for comparison

    # Remove batch dimension for plotting
    predicted_snapshot = predicted_snapshot.squeeze(0).cpu().numpy()  # Shape: [3, height, width]
    true_snapshot = true_snapshot.squeeze(0).cpu().numpy()  # Shape: [3, height, width]
    error_snapshot = np.abs(true_snapshot - predicted_snapshot)  # Absolute error

    # Plot the comparison for each component (u_x, u_y, p)
    components = ['u_x', 'u_y', 'p']
    for i, component in enumerate(components):
        plt.figure(figsize=(15, 5))

        # Ground Truth
        plt.subplot(1, 3, 1)
        plt.imshow(true_snapshot[i], cmap="viridis")
        plt.colorbar()
        plt.title(f"Ground Truth: {component}")

        # LSTM Prediction
        plt.subplot(1, 3, 2)
        plt.imshow(predicted_snapshot[i], cmap="viridis")
        plt.colorbar()
        plt.title(f"LSTM Prediction: {component}")

        # Error
        plt.subplot(1, 3, 3)
        plt.imshow(error_snapshot[i], cmap="inferno")
        plt.colorbar()
        plt.title(f"Absolute Error: {component}")

        plt.tight_layout()
        plt.show()

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
    plot_after=5  # Number of recursive predictions to wait before plotting
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
        input_sequence = initial_latents.unsqueeze(0)  # Shape: [1, seq_length, latent_dim]

        # Predict the next latent vector
        with torch.no_grad():
            next_latent = lstm(input_sequence, re_value_tensor)  # Shape: [1, 1, latent_dim]
            predicted_last_timestep = next_latent[:, -1, :]  # Shape: [batch_size, latent_dim]
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
            ground_truth_dir, f"{re_value}-data-{101.607+ 0.077 * (step + 1):.3f}"
        )
        ground_truth_field = load_data(ground_truth_file).detach().cpu().numpy()

        # Calculate and log MSE
        mse = np.mean((predicted_field_np - ground_truth_field) ** 2)
        print(f"Step {step + 1}: MSE with ground truth: {mse}")

        # Plot after the specified number of steps
        if (step + 1) % plot_after == 0 or (step + 1) == num_predictions:
            # components = ["u_x", "u_y", "p"]
            components = ["u_x"]
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


# decode_and_plot_comparison(file_path, encoder_path, decoder_path, device="cuda")
# plot_lstm_prediction(sequence_path, encoder_path, decoder_path, lstm_path, device="cuda")

# """
recursive_validation_with_plots(
    encoder_path,
    decoder_path,
    lstm_path,
    file_paths=sequence_path,  # Initial snapshots
    re_value=re_val,  # Normalized Reynolds number
    num_predictions=100,  # Number of recursive predictions
    ground_truth_dir="data/",
    output_dir="predicted_snapshots",
    device="cuda",
    plot_after=1  # Generate plots after every 5 predictions
)
# """
"""
average_loss = four_fold_cross_validation(
    model_class=CVAutoencoder,
    dataset=dataset,
    num_epochs=Epochs,  # Example epoch count
    batch_size=BatchSize,  # Adjust as needed
    learning_rate=1e-3,
    device=device
)
"""

"""
trained_lstm = train_lstm(
    lstm_model=ae.lstm,
    dataset=dataset,
    encoder=ae.encoder,
    device=device,
    latent_dim=new_latent_dimension,  # Match your encoder's latent dimension
    batch_size=BatchSize,
    epochs=Epochs,
    learning_rate=1e-4
)
"""

"""
recursive_validation(
    encoder_path=encoder_path,
    decoder_path=decoder_path,
    lstm_path=lstm_path,
    file_paths=init_file_paths,
    re_value=re_norm,
    num_predictions=num_predictions,
    ground_truth_dir="data/",
    device=device
)
"""

"""
# Save or process the predicted flow fields
for idx, field in enumerate(predicted_flow_fields):
    print(f"Prediction {idx + 1} has shape: {field.shape}")
    # Optionally save the flow field to disk
"""