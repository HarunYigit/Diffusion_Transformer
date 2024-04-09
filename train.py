from matplotlib import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import model
import torchvision.transforms as transforms

# Örnek bir veri kümesi sınıfı oluştur
class CustomDataset(Dataset):
    def __init__(self, input_folder, output_folder, transform=None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.input_files = os.listdir(input_folder)
        self.output_files = os.listdir(output_folder)
        self.transform = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_folder, self.input_files[idx])
        output_path = os.path.join(self.output_folder, self.output_files[idx])
        input_image = cv2.imread(input_path)
        output_image = cv2.imread(output_path)
        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)
        return input_image, output_image


# Örnek bir dönüştürme fonksiyonu tanımla (boyutları ayarla, normalleştirme yap vb.)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
])

# Veri kümesi ve veri yükleyici oluştur
input_folder = './inputs'
output_folder = './outputs'
dataset = CustomDataset(input_folder, output_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Modeli ve kayıp fonksiyonunu tanımla
model = model.model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim döngüsü
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Eğitim tamamlandıktan sonra modeli kaydedebilirsiniz
torch.save(model.state_dict(), 'input_diffusion_model.pth')
