import glob
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Subset, DataLoader
from process import  PretrainFlowDataset, SequenceDataset
from cae import CVAutoencoder, BatchSize, new_latent_dimension

Epochs = 1500
sequence_length = 21

encoder_path = "weights/encoder_weights.pth"
decoder_path = "weights/decoder_weights.pth"
lstm_path = "weights/lstm_weights.pth"

file_paths = sorted(glob.glob("data/training_data/*-data-*"))

re_numbers = [int(file.split('/')[1].split('-')[0]) for file in file_paths]
mean_re = np.mean(re_numbers)
std_re = np.std(re_numbers)

re_normalize = [(re - mean_re) / std_re for re in re_numbers]

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
ae = CVAutoencoder().to(device)

print("Processing data")

###### Only call one dataset function to save time ######
cae_dataset = PretrainFlowDataset(file_paths, re_normalize)
lstm_dataset = SequenceDataset(file_paths, re_normalize, sequence_length)

print("Processing finished")

def four_fold_cross_validation(model_class, dataset, num_epochs=100, batch_size=32, learning_rate=1e-4, device="cuda"):

    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    fold_results = []

    loss_history = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

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
        
        val_loss /= len(val_loader)
        print(f"Validation Loss for Fold {fold + 1}: {val_loss:.4f}")
        fold_results.append(val_loss)

    with open("CAE_loss_history.csv", "w", newline="") as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(["Epoch", "Loss"])
      for epoch, loss in enumerate(loss_history, 1):
        writer.writerow([epoch, loss])

    # Calculate and return the average validation loss across all folds
    average_loss = np.mean(fold_results)
    print(f"Average Validation Loss Across Folds: {average_loss:.4f}")
    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    print(f"Saved encoder and decoder for Fold {fold + 1}.")
    return average_loss

def train_lstm(lstm_model, device, batch_size, epochs, learning_rate=1e-3):

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
    
    precomputed_latents, precomputed_re_values = load_precomputed_latents("precomputed_latents.pt")
    dataset0 = list(zip(precomputed_latents, precomputed_re_values))

    # Prepare optimizer and loss function
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.9, patience=30)

    loss_history = []  
    val_loss_history = []

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

        scheduler.step(average_val_loss)

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

###### Uncomment to train CAE ######
"""
average_loss = four_fold_cross_validation(
    model_class=CVAutoencoder,
    dataset=cae_dataset,
    num_epochs=Epochs,  # Example epoch count
    batch_size=BatchSize,  # Adjust as needed
    learning_rate=1e-3,
    device=device
)
"""

###### Uncomment to train LSTM ######
"""
### Comment out after running once to save a lot of time ###
precompute_latent_vectors(lstm_dataset, ae.encoder, device, new_latent_dimension, save_path="precomputed_latents.pt")

trained_lstm = train_lstm(
    lstm_model=ae.lstm,
    device=device,
    batch_size=BatchSize,
    latent_dim=new_latent_dimension,  # Match your encoder's latent dimension
    epochs=Epochs,
    learning_rate=1e-4
)
"""
