import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def load_data(file_path):
    # Step 1: Read the text file
    data = np.loadtxt(file_path, skiprows=1)  # Skip the header row

    # Step 2: Extract the relevant columns
    x = data[:, 0]    # x-coordinates (you may not need this for the network)
    y = data[:, 1]    # y-coordinates (you may not need this for the network)
    u_x = data[:, 2]  # u_x velocity component
    u_x /= np.max(np.abs(u_x))
    u_y = data[:, 3]  # u_y velocity component
    u_y /= np.max(np.abs(u_y))
    p = data[:, 4]    # pressure
    p /= np.max(np.abs(p))

    # Step 3: Calculate the actual grid dimensions
    x_unique = len(np.unique(x))
    y_unique = len(np.unique(y))

    # Step 4: Reshape the data into (y_unique, x_unique)
    u_x = u_x.reshape(y_unique, x_unique)
    u_y = u_y.reshape(y_unique, x_unique)
    p = p.reshape(y_unique, x_unique)

    # Step 4: Stack the channels (u_x, u_y, p) together as needed for NN input
    # Stack into a 3-channel input: [u_x, u_y, p]
    input_data = np.stack([u_x, u_y, p], axis=0)  # Shape: [3, height, width]
    # print(input_data.shape)
    # Step 5: Convert to a PyTorch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    # plot_tensor(input_tensor, x, y)

    return input_tensor

def load_all_snapshots(file_paths):
    """
    Load all snapshots from the given file paths into a single tensor of shape 
    [# of inputs, 3, height, width].

    Args:
        file_paths: List of file paths containing the snapshot data.

    Returns:
        all_snapshots: Tensor of shape [# of inputs, 3, height, width].
    """
    all_snapshots = []

    for file_path in file_paths:
        # Load the data for a single snapshot
        data = np.loadtxt(file_path, skiprows=1)  # Skip the header row

        # Extract the relevant columns
        u_x = data[:, 2]  # u velocity
        u_y = data[:, 3]  # v velocity
        p = data[:, 4]    # pressure

        # Normalize each field (optional, based on your original code)
        u_x /= np.max(np.abs(u_x))
        u_y /= np.max(np.abs(u_y))
        p /= np.max(np.abs(p))

        # Determine grid dimensions
        x_unique = len(np.unique(data[:, 0]))  # x-coordinates
        y_unique = len(np.unique(data[:, 1]))  # y-coordinates

        # Reshape each field into [height, width]
        u_x = u_x.reshape(y_unique, x_unique)
        u_y = u_y.reshape(y_unique, x_unique)
        p = p.reshape(y_unique, x_unique)

        # Stack into [channels, height, width] format
        snapshot = np.stack([u_x, u_y, p], axis=0)

        # Append to the list
        all_snapshots.append(snapshot)

    # Convert list of arrays into a single NumPy array before tensor conversion
    all_snapshots = np.array(all_snapshots)  # Shape: [# of inputs, 3, height, width]

    # Convert to a PyTorch tensor
    all_snapshots = torch.tensor(all_snapshots, dtype=torch.float32)  # Shape: [# of inputs, 3, height, width]

    return all_snapshots

class PretrainFlowDataset(Dataset):
    def __init__(self, file_paths, re_numbers):
        self.data = [load_data(file_path) for file_path in file_paths]
        self.re_numbers = re_numbers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        snapshot = self.data[idx]  # Single snapshot
        re_value = torch.tensor(self.re_numbers[idx], dtype=torch.float32).unsqueeze(0)  # Shape: [1]
        return snapshot, re_value


class SequenceDataset(Dataset):
    def __init__(self, file_paths, re_numbers, sequence_length):
        """
        Args:
            file_paths: List of all input files to convert to [num_snapshots, channels, height, width].
            re_numbers: List of corresponding Reynolds numbers for each snapshot.
            sequence_length: The length of each sequence for the LSTM.
        """
        self.snapshot_data = load_all_snapshots(file_paths)  # Shape: [# of snapshots, 3, 256, 256]
        self.re_numbers = torch.tensor(re_numbers, dtype=torch.float32)
        self.sequence_length = sequence_length

        self.sequences = []
        self.re_labels = []

        # Group snapshots by Reynolds number
        unique_re = torch.unique(self.re_numbers)
        for re in unique_re:
            mask = self.re_numbers == re
            snapshots = self.snapshot_data[mask]

            # Generate sequences for this Re
            num_snapshots = snapshots.size(0)
            for i in range(num_snapshots - sequence_length + 1):
                # print(f"Creating sequences for Re: {re} # {i}")
                sequence = snapshots[i : i + sequence_length]  # Shape: [sequence_length, 3, height, width]
                self.sequences.append(sequence)
                self.re_labels.append(re)

        # Normalize (is this necessary anymore?)
        #mean_re = np.mean(re_numbers)
        #std_re = np.std(re_numbers)
        #self.re_labels = [(re - mean_re) / std_re for re in re_numbers]

        # Convert to tensors
        self.sequences = torch.stack(self.sequences)  # Shape: [# of sequences, sequence_length, 3, height, width]
        self.re_labels = torch.tensor(self.re_labels, dtype=torch.float32)  # Shape: [# of sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]  # Shape: [sequence_length, 3, height, width]
        re_value = self.re_labels[idx]  # Scalar Re value
        #print(f"Sequence index: {idx}, Re: {re_value}")
        #print(f"First snapshot in sequence: {sequence[0, :, 0, 0]}")  # Example snapshot data
        return sequence, re_value
