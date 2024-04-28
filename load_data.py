import os
import torch
import numpy as np
import imageio

if not os.path.exists("data_folders"):
    os.mkdir("data_folders")
if not os.path.exists("data_folders/train"):
    os.mkdir("data_folders/train")
if not os.path.exists("data_folders/eval"):
    os.mkdir("data_folders/eval")