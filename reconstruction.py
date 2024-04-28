import os
import torch
import numpy as np
from pythae.models import VAE
from pythae.models import AutoModel
from pythae.samplers import NormalSampler
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image
from pythae.data.datasets import DatasetOutput

class TrainColorizerDataset(datasets.ImageFolder):
    def __init__(self, color_dir, gray_dir, transform=None):
        self.color_dir = color_dir
        self.gray_dir = gray_dir
        self.transform = transform
        self.color_files = os.listdir(color_dir)
        self.gray_files = os.listdir(gray_dir)

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, index):
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

# datasets
color_dir = 'eval/color'
gray_dir = 'eval/gray'

data_transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

eval_dataset = TrainColorizerDataset(color_dir, gray_dir, transform=data_transform)

trained_model = AutoModel.load_from_folder('models/4')
device = "cpu" if torch.cuda.is_available() else "cpu"

batch_size = 64
grayscale_images = []
for i in range(batch_size):
    index = np.random.randint(len(eval_dataset))
    grayscale_image = eval_dataset[index]['data']
    grayscale_images.append(grayscale_image)



grayscale_images = torch.stack(grayscale_images)

# Reconstruct the images
reconstructed_images = trained_model.reconstruct(grayscale_images).detach()
#reconstructed_images = grayscale_images


fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(15, 8))
for i in range(4):
    for j in range(8):
        index = i * 8 + j
        ax = axes[i, j]
        ax.imshow(reconstructed_images[index].permute(1, 2, 0))
        ax.axis('off')
plt.tight_layout()
plt.show()