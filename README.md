# Convolutional Autoencoder with LSTM for CFD Predictions

This repository contains a machine learning model that combines a **Convolutional Autoencoder (CAE)** and **Long-Short Term Memory (LSTM)** network to predict unsteady flow fields around a two-dimensional cylinder, similar to that done by Hasegawa, K., et al. (2020). The model is trained on **Computational Fluid Dynamics Simulations (CFD)** using Basilisk.

## Features
- Predicts velocity and pressure fields for various Reynolds numbers.
- Integrates CAE for dimensionality reduction and reconstruction.
- Uses LSTM to capture temporal dynamics for short-term predictions.
- Trained on snapshots of flow fields for steady and unsteady regimes.

## Project Structure
```
├── data
│   ├── training_data/
│   └── validation_data/
├── docs
│   ├── project_proposal
│   ├── final_presentation.pdf
│   └── final_report.pdf
├── weights
│   ├── encoder_weights.pth
│   ├── decoder_weights.pth
│   └── lstm_weights.pth
├── latent_vectors
│   ├── latent_predictions_*.npy
│   └── latent_inputs_*.npy
├── sim
│   ├── ibmcylinder 
│   │   └── simulation data
│   ├── Makefile
│   ├── ibm*.h
│   └── ibmclylinder.c
├── process.py
├── cae.py
├── train.py
├── validate.py
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Basilisk (to run simulations)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/csce585-mlsystems/cae_cylinder.git
   cd cae_cylinder
   ```
2. Install dependencies:
   ```bash, e.g.
   pip3 install torch
   ```
2.1 Install Basilisk (if desired):
   ```bash, 
   darcs clone http://basilisk.fr/basilisk
   ```
   More information here: http://basilisk.fr/src/INSTALL

### Usage

#### 0. Run Simulations (to obtain data)
The simulation data was too large to upload to GitHub. To generate this simulation data, go to the simulation folder.
```bash
cd sim
```
and run the simulation file
```
make ibmcylinder.tst
```
which can be modified by 
```
vim ibmcylinder.c
```
and changing Re to the desired value (default is 100)
``` c
int main() {
  ...
  Re = CHANGE_ME;
  run();
}
```

#### 1. Train the CAE
Train the Convolutional Autoencoder to reconstruct flow fields. Assuming all of the data for training is stored in **data/training_data**.

Uncomment the **cae_dataset** line in train.py like so
``` python
###### Only call one dataset function to save time ######
cae_dataset = PretrainFlowDataset(file_paths, re_normalize)
# lstm_dataset = SequenceDataset(file_paths, re_normalize, sequence_length)
```
and uncomment the function call to train the CAE
``` python
###### Uncomment to train CAE ######
average_loss = four_fold_cross_validation(
    model_class=CVAutoencoder,
    dataset=cae_dataset,
    num_epochs=Epochs,  # Example epoch count
    batch_size=BatchSize,  # Adjust as needed
    learning_rate=1e-3,
    device=device
)
```
A batch size of 512 was used to train the CAE, although this can be adjusted depending on your machine.

Run the file
```bash
python3 train.py
```

#### 2. Train the LSTM
To train the LSTM, make sure only **lstm_dataset** set is uncommented
``` python
###### Only call one dataset function to save time ######
# cae_dataset = PretrainFlowDataset(file_paths, re_normalize)
lstm_dataset = SequenceDataset(file_paths, re_normalize, sequence_length)
```
and the **train_lstm function** call is uncommented
``` python
###### Uncomment to train LSTM ######
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
```
I recommend commenting out the **precompute_latent_vectors** after the first time you run it to save startup time if you decide to retrain the LSTM w/o changing the latent spaces.

Run the file
```bash
python3 train.py
```

#### 3. Validate CAE Results

To test how well the CAE can reconstruct flow fields, simply specify the desired data file
''' python
file_path = "data/40-data-100.375"  # file to test CAE reconstruction
'''
and uncomment out the function call
```python
###### Uncomment to test CAE given a file (file_path) ######
decode_and_plot_comparison(file_path, encoder_path, decoder_path, device="cuda")
```
and run
```bash
python3 validate.py
```
#### 4. Validate CAE w/LSTM Results
Make sure to comment out the **decode_and_plot_comparison** function call done in step 3.

Include the file paths of the initial snapshots you want to input into the LSTM (can be any number)
```python
sequence_path = [                   # specify what files to input to test LSTM prediction (can be any #)
    "validate/180-data-100.067",
    "validate/180-data-100.144",
     ...
    "validate/180-data-101.530",
    "validate/180-data-101.607"
]
```

Specify the Reynolds number (Re) that you wish to input into the LSTM and the last time in the given sequence, e.g.
```python
startTime = 101.607
dt = 0.077                          # time interval between snapshots
RE = 180
```
Finally, uncomment out the **recursive_validation_with_plots** function call
```python
##### Uncomment to test CAE w/LSTM given a sequence (sequence_path) ######
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
```
and specify the number of predictions and how often to generate plots, default is 
```python
plot_after=10,  # Generate plots after every 10 predictions
```
and run
```bash
python3 visualize/validate.py
```

#### 5. Plot latent spaces
You can plot the predicted and target latent spaces using PCA and t-SNE analysis methods using the scripts found in the **visualize/** directory. A few sample latent spaces are already provided in **latent_vectors**, so you can just run the files like
```bash
python3 pca.py
```
or
```bash
python3 t-sne.py
```
which will make a pop-up window of the plotted results.

## Results
- The CAE achieves low reconstruction error on trained and most untrained Reynolds numbers.
- The LSTM captures short-term dynamics but requires improvements for long-term stability.
- Example results are available in the `results` directory.
- Video presentation: https://youtu.be/s3koj2zgMiE
- Final presentation: docs/final_presentation.pdf
- Final report: docs/final_report.pdf

## Future Work
- Enhance the LSTM architecture for long-term predictions.
- Optimize the latent space dimensionality for better feature representation.
- Explore variational autoencoders (VAEs) for improved data normalization.

## References
- Basilisk CFD Solver: [basilisk.fr](http://basilisk.fr)
- Hasegawa, K., et al. (2020). CNN-LSTM Based Reduced Order Modeling.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to the University of South Carolina's Computational Thermo-Fluid Laboratory and Dr. Pooyan Jamshidi for supporting this work.
