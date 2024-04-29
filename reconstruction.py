"""
Colorization Evaluation Script
Author: Justin Lin
"""

import os
import sys
import torch
import numpy as np
from pythae.models import AutoModel
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image
from pythae.data.datasets import DatasetOutput

class TrainColorizerDataset(datasets.ImageFolder):
    """
    Dataset class for colorization evaluation.
    """

    def __init__(self, color_dir, gray_dir, transform=None):
        """
        Initialize the dataset.

        Args:
            color_dir (str): Path to directory containing color images.
            gray_dir (str): Path to directory containing grayscale images.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.color_dir = color_dir
        self.gray_dir = gray_dir
        self.transform = transform
        self.color_files = os.listdir(color_dir)
        self.gray_files = os.listdir(gray_dir)

    def __len__(self):
        """
        Return the number of images in the dataset.
        """
        return len(self.color_files)

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            DatasetOutput: An object containing the grayscale image as data and the color image as target.
        """
        color_file = os.path.join(self.color_dir, self.color_files[index])
        gray_file = os.path.join(self.gray_dir, self.gray_files[index])

        color_img = Image.open(color_file).convert('RGB')
        gray_img = Image.open(gray_file).convert('L')

        if self.transform:
            color_img = self.transform(color_img)
            gray_img = self.transform(gray_img)

        return DatasetOutput(
            data=gray_img,
            target=color_img
        )

def main(model_path):
    # Dataset paths
    color_dir = 'eval/color'
    gray_dir = 'eval/gray'

    # Image transformations
    data_transform = transforms.Compose([
        transforms.Resize((128, 128), antialias=True),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    # Load evaluation dataset
    eval_dataset = TrainColorizerDataset(color_dir, gray_dir, transform=data_transform)

    # Load pre-trained model
    trained_model = AutoModel.load_from_folder(model_path)
    device = "cpu" if torch.cuda.is_available() else "cpu"

    # Batch processing
    batch_size = 64
    grayscale_images = []
    for i in range(batch_size):
        index = np.random.randint(len(eval_dataset))
        grayscale_image = eval_dataset[index]['data']
        grayscale_images.append(grayscale_image)

    grayscale_images = torch.stack(grayscale_images)

    # Reconstruct images
    reconstructed_images = trained_model.reconstruct(grayscale_images).detach()

    # Visualization
    num_images = 4
    num_rows = num_images
    num_cols = 4

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, 4 * num_rows))

    for i in range(num_images):
        # Display grayscale image
        ax = axes[i, 0]
        ax.imshow(grayscale_images[i].permute(1, 2, 0))
        ax.axis('off')
        
        # Display reconstructed image
        ax = axes[i, 1]
        ax.imshow(reconstructed_images[i].permute(1, 2, 0))
        ax.axis('off')
        
        # Display additional grayscale image
        ax = axes[i, 2]
        ax.imshow(grayscale_images[i + 4].permute(1, 2, 0))
        ax.axis('off')
        
        # Display additional reconstructed image
        ax = axes[i, 3]
        ax.imshow(reconstructed_images[i + 4].permute(1, 2, 0))
        ax.axis('off')

    # Adjust subplot spacing
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reconstruction.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    main(model_path)