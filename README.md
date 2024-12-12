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
│   └── final_report.pdf
├── weights
│   ├── cae_model.pth
│   └── lstm_model.pth
├── latent_vectors
│   ├── latent_predictions_*.npy
│   └── latent_inputs_*.npy
├── process.py
├── cae.py
├── train.py
├── validate.py
│   └── predict.py

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
   '''bash, 
   darcs clone http://basilisk.fr/basilisk
   '''
### Usage

#### 0. Run Simulations (if desired)
Prepare the training and validation datasets from raw simulation snapshots.
```bash
pyscripts/preprocess_data.py
```

#### 2. Train the CAE
Train the Convolutional Autoencoder to reconstruct flow fields.
```bash
python scripts/train_cae.py
```

#### 3. Train the LSTM
Train the LSTM to predict time-series data in the latent space.
```bash
python scripts/train_lstm.py
```

#### 4. Make Predictions
Generate predictions for new flow field data.
```bash
python scripts/predict.py
```

## Results
- The CAE achieves low reconstruction error on trained Reynolds numbers.
- The LSTM captures short-term dynamics but requires improvements for long-term stability.
- Example results are available in the `results` directory.

## Future Work
- Enhance the LSTM architecture for long-term predictions.
- Optimize the latent space dimensionality for better feature representation.
- Explore variational autoencoders (VAEs) for improved data normalization.

## References
- Basilisk CFD Solver: [basilisk.fr](http://basilisk.fr)
- Hasegawa, K., et al. (2020). CNN-LSTM Based Reduced Order Modeling.
- Bank, D., et al. (2020). Autoencoders.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to the University of South Carolina's Computational Thermo-Fluid Laboratory for supporting this work.
