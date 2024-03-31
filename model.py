import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import os
import cv2
import numpy as np

class InputDiffusionTransformer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(InputDiffusionTransformer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # 0-1 aralığında piksel değerleri için sigmoid aktivasyonu
        )

    def forward(self, x):
        # Encoder'a giriş görüntüsünü geçir
        encoded = self.encoder(x)
        # Decoder'dan çıkış görüntüsünü üret
        decoded = self.decoder(encoded)
        return decoded
    
input_channels = 3  # RGB görüntü için 3 kanal
output_channels = 3  # Çıktı olarak RGB görüntü için 3 kanal
model = InputDiffusionTransformer(input_channels, output_channels)
batch_size = 1  # Batch boyutu 1 olarak varsayıldı
input_image = torch.randn(batch_size, input_channels, 256, 256)  # Örnek bir giriş görüntüsü (boyutlar örnek olarak verildi)

# Modeli kullanarak çıkış görüntüsünü üret
output_image = model(input_image)

# NumPy dizisine dönüştür
output_image_np = output_image.detach().numpy()  # Detach işlemi gradient hesaplamalarını kaldırır

# Çıkış görüntüsünü OpenCV ile göster
output_image_np = np.transpose(output_image_np.squeeze(), (1, 2, 0))  # Kanal boyutlarını değiştir ve gereksiz boyutları kaldır
output_image_np = (output_image_np * 255).astype(np.uint8)  # 0-1 aralığını 0-255 aralığına dönüştür ve uint8 veri tipine çevir
input_image_np = input_image.detach().numpy()
input_image_np = np.transpose(input_image_np.squeeze(), (1, 2, 0))  # Kanal boyutlarını değiştir ve gereksiz boyutları kaldır
input_image_np = (input_image_np * 255).astype(np.uint8)  # 0-1 aralığını 0-255 aralığına dönüştür ve uint8 veri tipine çevir

print(output_image_np.shape)
# cv2.imshow('Output Image', cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR))
# cv2.imshow('Input Image', cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()