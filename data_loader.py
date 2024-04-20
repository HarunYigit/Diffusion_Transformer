import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from torch.utils.data import random_split
from utils import flatten_image,preprocess_image,split_image_into_patches
import numpy as np
import json
class CustomDataset(Dataset):
    def __init__(self, input_folder,  transform=None):
        self.input_folder = input_folder
        self.input_files = os.listdir(input_folder)
        self.transform = transform

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
config = json.load(open("config.json"))
batch_size= config['batch_size']
dataset = CustomDataset(input_folder)

# Veri setini train ve test olarak ayır
train_size = int(0.8 * len(dataset))  # %80 train, %20 test
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Dataloaders oluştur
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Train size:",len(train_dataloader) * batch_size)
print("Test size:",len(test_dataloader) * batch_size)