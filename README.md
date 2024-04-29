# VAE Image Colorization

This project involves training a Variational Autoencoder (VAE) for image colorization and evaluating its performance. The VAE architecture is implemented using PyTorch and the Pythae library. The trained model takes grayscale images as input and generates colorized versions as output.

## Files Included:

1. **train_colorizer.py**: This script contains the training pipeline for the VAE model. It includes:
   - Definition of the VAE architecture (`Encoder_Conv_VAE_ColorImage` and `Decoder_Conv_VAE_ColorImage` classes).
   - Custom dataset class for loading training data (`TrainColorizerDataset`).
   - Main function for training the model (`main` function).
   - Usage of argparse for specifying training parameters.

2. **reconstruction.py**: This script contains the evaluation pipeline for the trained VAE model. It includes:
   - Loading of the pre-trained model.
   - Definition of the evaluation dataset class (`TrainColorizerDataset`).
   - Main function for reconstructing color images from grayscale ones and visualizing the results (`main` function).
   - Usage of command-line arguments to specify the path to the pre-trained model.

3. **vae_model.py**: This file contains the definition of the VAE model class.

4. **base_trainer.py**: This file contains the base trainer class for training the VAE model.

## Dependencies:

To run the code successfully, make sure you have the following dependencies installed:

- Python (>=3.6)
- PyTorch
- Pythae
- NumPy
- Matplotlib
- Pillow (PIL)

You can install the required Python packages using the following command:

pip install torch pythae numpy matplotlib pillow

## Usage:

### Training:

To train the VAE model for image colorization, run the `trainer.py` script with the desired training parameters:

python trainer.py --num_epochs <num_epochs> --latent_dim <latent_dim>

- `<num_epochs>`: Number of epochs for training.
- `<latent_dim>`: Dimension of the latent space.

### Evaluation:

To evaluate the trained VAE model on grayscale images and visualize the reconstructed colorized versions, run the `reconstruction.py` script with the path to the pre-trained model:

python reconstruction.py <model_path>


- `<model_path>`: Path to the folder containing the model files.

Also included are oue models we trained for the experiment in the models folder.

## Author:

- [Justin Lin](https://github.com/JustinL15)

--- 
