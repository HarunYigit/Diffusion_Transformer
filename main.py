import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import os
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
    

input_folder = "./inputs"
output_folder = "./outputs"

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Giriş resimlerini belirli bir boyuta yeniden boyutlandırın
    transforms.ToTensor(),  # Resimleri tensorlere dönüştürün
])

dataset = ImageFolder(input_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = InputDiffusionTransformer().to(device)
criterion = nn.MSELoss()  # Örnek bir kayıp fonksiyonu kullanıldı, isteğe bağlı değiştirilebilir
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    for batch_idx, (inputs, _) in enumerate(dataloader):
        inputs = inputs.to(device)

        # Modelden çıktı alın
        outputs = model(inputs)

        # Kayıp hesaplama ve geriye doğru geçiş
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")

# Modeli kaydedin
torch.save(model.state_dict(), "input_diffusion_transformer.pth")
