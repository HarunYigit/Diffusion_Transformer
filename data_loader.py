import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from utils import flatten_image,preprocess_image,split_image_into_patches
import numpy as np
import json
class CustomDataset(Dataset):
    def __init__(self, input_folder,  transform=None, noise_level=0.1, num_iterations=2):
        self.input_folder = input_folder
        self.input_files = os.listdir(input_folder)
        self.transform = transform
        self.noise_level = noise_level
        self.num_iterations = num_iterations

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_folder, self.input_files[idx])

    
        input_image = preprocess_image(input_path)
        # patches = split_image_into_patches(input_image)
        input_image = flatten_image(input_image)
        # patches_tensor = torch.tensor(patches, dtype=torch.float32)
        
        return torch.from_numpy(input_image)


# Örnek bir dönüştürme fonksiyonu tanımla (boyutları ayarla, normalleştirme yap vb.)


# Veri kümesi ve veri yükleyici oluştur
input_folder = '.\\inputs'
dataset = CustomDataset(input_folder)
config = json.load(open("config.json"))
batch_size= config['batch_size']
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
