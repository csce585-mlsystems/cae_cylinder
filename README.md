# Convolutional Autoencoder with LSTM for CFD Predictions

This repository contains a machine learning model that combines a **Convolutional Autoencoder (CAE)** and **Long-Short Term Memory (LSTM)** network to predict unsteady flow fields around a two-dimensional cylinder. The model is trained on ***Computational Fluid Dynamics Simulations (CFD)*** using Basilisk.

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

#### 0. Run Simulations (if desired)
To generate new data, go to the simulation folder.
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
Train the Convolutional Autoencoder to reconstruct flow fields.
Assuming all of the data for training is stored in data/training_data
```bash
python scripts/train_cae.py
```



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
