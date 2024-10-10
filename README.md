
#  Convolutional Variational Autoencoder (CNN VAE) with MNIST

## Project Overview

This project involves the implementation of a **Convolutional Variational Autoencoder (VAE)** using PyTorch. The model is trained on the MNIST dataset to perform dimensionality reduction and image generation.

## Key Features
- Implementation of a CNN-based VAE.
- Utilizes the MNIST dataset for training and evaluation.
- The model learns a latent representation of the images and generates new images from the learned latent space.
- Includes training, validation, and loss computation routines.

## Files in the Repository
- **Msds23019_assignment2.ipynb**: The main Jupyter Notebook containing the code for data loading, model architecture, training loop, and evaluation.

## Model Architecture
- The model uses a Convolutional Neural Network (CNN) as the encoder and decoder.
- The latent space dimension is 20, which captures key features of the input images.
- The decoder reconstructs images from the latent space.

## How to Run
1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Open the Jupyter notebook:
    ```bash
    jupyter notebook Msds23019_assignment2.ipynb
    ```

## Dataset
- The project uses the **MNIST** dataset, which consists of 28x28 grayscale images of handwritten digits.
- The dataset is automatically downloaded from PyTorch's torchvision library.

## Dependencies
- Python 3.10+
- PyTorch
- Torchvision
- Matplotlib

You can install the required dependencies using the `requirements.txt` file.

## Training the Model
- The model is trained for 2 epochs with a learning rate of 0.0003 and a batch size of 64.
- The training and validation loss are computed at intervals, and the performance is evaluated using the reconstruction loss.

## Results
- The model generates new MNIST-like images from the latent space after training.
- It also reconstructs images from the input during the testing phase.

## Future Improvements
- Increase the number of training epochs for better convergence.
- Experiment with different latent space dimensions.
- Explore other datasets for generalization.

## Author
- Rabia Shakoor

## License
This project is licensed under the MIT License. See the LICENSE file for details.
# Variational-autoencoder
