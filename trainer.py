"""
File: trainer.py
Author: Justin Lin

Description: This script trains a Variational Autoencoder (VAE) for image colorization. It defines the VAE architecture,
             data loading pipeline, and training procedure.

Dependencies: Ensure you have the necessary dependencies installed, including PyTorch, torchvision, PIL, and pythae.

Usage:
    python train_colorizer.py --num_epochs [NUM_EPOCHS] --latent_dim [LATENT_DIM]

Parameters:
    --num_epochs: Number of epochs for training.
    --latent_dim: Dimension of the latent space.

Example:
    python train_colorizer.py --num_epochs 50 --latent_dim 64
"""


import os
import argparse
from PIL import Image
from pythae.models import VAEConfig
from vae_model import VAE
from base_trainer import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.data.datasets import DatasetOutput
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput


# Define the encoder architecture for the VAE
class Encoder_Conv_VAE_ColorImage(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = (3, 128, 128)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, 4, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(256 * 8 * 8, 16)
        self.embedding = nn.Linear(16, args.latent_dim)
        self.log_var = nn.Linear(16, args.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x)
        h1 = self.flatten(h1)
        h1 = self.dense(h1)
        
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        
        return output


# Define the decoder architecture for the VAE
class Decoder_Conv_VAE_ColorImage(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.input_dim = (3, 128, 128)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        self.fc = nn.Linear(args.latent_dim, 256 * 8 * 8)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.n_channels, 3, 1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        h1 = self.fc(z).reshape(z.shape[0], 256, 8, 8)
        output = ModelOutput(reconstruction=self.deconv_layers(h1))
        return output

class TrainColorizerDataset(datasets.ImageFolder):
    def __init__(self, color_dir, gray_dir, transformC=None, transformG=None):
        self.color_dir = color_dir
        self.gray_dir = gray_dir
        self.transformC = transformC
        self.transformG = transformG
        self.color_files = os.listdir(color_dir)
        self.gray_files = os.listdir(gray_dir)

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, index):
        color_file = os.path.join(self.color_dir, self.color_files[index])
        gray_file = os.path.join(self.gray_dir, self.gray_files[index])

        color_img = Image.open(color_file).convert('RGB')
        gray_img = Image.open(gray_file).convert('RGB')

        if self.transformC:
            color_img = self.transformC(color_img)
            gray_img = self.transformG(gray_img)

        return DatasetOutput(
            data=gray_img,
            target=color_img
        )

def main(args):
    # Set up paths and data loading
    color_dir = 'train/color'
    gray_dir = 'train/gray'

    data_transformC = transforms.Compose([
        transforms.Resize((128, 128), antialias=True),
        transforms.ToTensor()
    ])

    data_transformG = transforms.Compose([
        transforms.Resize((128, 128), antialias=True),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    ecolor_dir = 'eval/color'

    train_dataset = TrainColorizerDataset(color_dir, color_dir, transformC=data_transformC, transformG=data_transformG)
    eval_dataset = TrainColorizerDataset(ecolor_dir, ecolor_dir, transformC=data_transformC, transformG=data_transformC)

    model_config = VAEConfig(
        input_dim=(3, 128, 128),
        latent_dim=args.latent_dim,
        reconstruction_loss="mse"
    )

    training_config = BaseTrainerConfig(
        output_dir='models',
        learning_rate=1e-3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_epochs=args.num_epochs,
        optimizer_cls="AdamW",
        optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.99)},
        no_cuda=False
    )
    encoder = Encoder_Conv_VAE_ColorImage(model_config)
    decoder = Decoder_Conv_VAE_ColorImage(model_config)

    vae_model = VAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder
    )

    pipeline = TrainingPipeline(
        training_config=training_config,
        model=vae_model
    )

    # initialize training
    pipeline(
        train_data=train_dataset,
        eval_data=eval_dataset
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE for image colorization")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs for training")
    parser.add_argument("--latent_dim", type=int, help="Dimension of the latent space")
    args = parser.parse_args()

    if not (args.num_epochs and args.latent_dim):
        parser.print_usage()
        exit()

    main(args)