import torch
import torch.nn as nn
import numpy as np

num_kernel = 5  # filter
num_stride = 2
num_padding = 4
lstm_layers = 7  # LSTM layers

latent_dimension = 82944
new_latent_dimension = 512
lstm_hidden_dim = 512
init_channels = 3       # u,v,p
layer1_channels = 16

BatchSize = 64

class Encoder (nn.Module):
    def __init__(self, channel_in=init_channels, num_channels=layer1_channels, z_dim=latent_dimension):
        super(Encoder, self).__init__()
        
        activation = nn.ReLU()

        # input: 3x256x256
        self.encoder = nn.Sequential(
            nn.Conv2d(channel_in, num_channels, num_kernel, stride=num_stride, padding=num_padding),
            nn.BatchNorm2d(num_channels),
            activation,

            nn.Conv2d(num_channels, num_channels*2, num_kernel, stride=num_stride, padding=num_padding),
            nn.BatchNorm2d(num_channels*2),
            activation,

            nn.Conv2d(num_channels*2, num_channels*4, num_kernel, stride=num_stride, padding=num_padding),
            nn.BatchNorm2d(num_channels*4),
            activation,

            nn.Flatten(),

            nn.Linear(latent_dimension, new_latent_dimension),
            activation,
            nn.BatchNorm1d(new_latent_dimension)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class LSTM (nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTM, self).__init__()

        self.latent_size = latent_dim
        self.hidden_size = hidden_dim
        
        torch.backends.cudnn.enabled = True
        self.lstm = nn.LSTM(latent_dim + 1, hidden_dim, num_layers, batch_first=True, dropout=dropout) # + 1 for Re
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latent_sequence, re_value):
        batch_size, sequence_length, _ = latent_sequence.size()
        re_expanded = re_value  # [batch_size, sequence_length, 1]

        # Concatenate latent vectors with Re
        lstm_input = torch.cat([latent_sequence, re_expanded], dim=-1)  # [batch_size, sequence_length, latent_dim + 1]

        # LSTM forward pass
        lstm_output, _ = self.lstm(lstm_input)  # lstm_output: [batch_size, sequence_length, hidden_dim]

        # Pass through the fully connected layer to predict the next latent vector
        output = self.fc(lstm_output)  # Shape: [batch_size, latent_dim]

        return output  # Shape: [batch_size, 1, latent_dim]
    

class Decoder (nn.Module):
    def __init__(self, channel_in=init_channels,num_channels=layer1_channels, z_dim=new_latent_dimension):
        super(Decoder, self).__init__()
        
        activation = nn.ReLU()

        self.decoder = nn.Sequential(
            nn.Linear(new_latent_dimension, latent_dimension),
            nn.Unflatten(1, torch.Size([layer1_channels*4, 36, 36])),

            nn.BatchNorm2d(num_channels*4),
            activation,

            nn.ConvTranspose2d(num_channels*4, num_channels*2, num_kernel, stride=num_stride, padding=num_padding, output_padding=0),
            nn.BatchNorm2d(num_channels*2),
            activation,

            nn.ConvTranspose2d(num_channels*2, num_channels, num_kernel, stride=num_stride, padding=num_padding, output_padding=1),
            nn.BatchNorm2d(num_channels),
            activation,

            nn.ConvTranspose2d(num_channels, channel_in, num_kernel, stride=num_stride, padding=num_padding, output_padding=1),
            nn.BatchNorm2d(channel_in)
        )

    def forward(self, x):
        return self.decoder(x)
    

class CVAutoencoder (nn.Module):
    def __init__(self, channel_in=init_channels, num_channels=layer1_channels, z_dim=new_latent_dimension):
        super(CVAutoencoder, self).__init__()

        # self.re = Re()
        self.encoder = Encoder(channel_in, num_channels, z_dim)
        self.lstm = LSTM(z_dim, lstm_hidden_dim, lstm_layers)
        self.decoder = Decoder(channel_in, num_channels, z_dim)

    def encode(self, x):
        return self.encoder(x)
    
    def rnn(self, x):
        return self.lstm(x)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, re_value):
        latent_sequence = self.encode(x)
        reconstructed_sequence = self.decode(latent_sequence)
        return reconstructed_sequence
