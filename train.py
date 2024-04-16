from matplotlib import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import model
import torchvision.transforms as transforms
import utils
import numpy as np
def flatten_image(image):
    flatten = []
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            flatten.append(image[y,x])
    flatten = np.array(flatten)
    return flatten
def reshape(flatten):
    image =np.zeros((28,28))
    x = 0
    y = 0
    for i in flatten:
        image[y,x] = i
        y += 1  # Y indeksi artırılıyor
        if y == 28:  # Y indeksi 28 olduğunda (satır bittiğinde), X indeksini artır ve Y'yi sıfırla
            x += 1
            y = 0
    return np.array(image)
# Örnek bir veri kümesi sınıfı oluştur
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
        noise = int(input_path[input_path.find("_noise_")+7:input_path.find(".",input_path.find("_noise_")+7)])

        output_path = input_path.replace("_noise_" + str(noise),"_noise_" + str(noise-1))
    
        input_image = cv2.imread(input_path)
    
        output_image = None
        if noise == 74:
            output_image = input_image.copy()
            input_image = cv2.imread(output_path.replace("\\noises","\\lastest_noises"))
        elif noise == 1:
            return self.__getitem__(idx+1)
        else:
            output_image = cv2.imread(output_path)
        # print(input_image,)
        input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY) 
        output_image = cv2.cvtColor(output_image,cv2.COLOR_BGR2GRAY) 

        input_image = flatten_image(input_image).astype(np.float32)
        output_image = flatten_image(output_image).astype(np.float32) / 255

        # print(input_image[150:155])

        # image = cv2.resize(input_image,(28,28))
        # image2 = cv2.resize(output_image,(28,28))
        image = reshape(input_image)
        image2 = reshape(output_image)
        input_image = torch.from_numpy(input_image)
        output_image = torch.from_numpy(output_image)
        # print(input_image.shape,output_image.shape,"shape")
        # cv2.imshow("asd",cv2.resize(image,(280,280)))
        # cv2.imshow("asdf",cv2.resize(image2,(280,280)))
        # cv2.waitKey(0)
        # print(input_image.shape,output_image.shape,"shape")
        return input_image, output_image


# Örnek bir dönüştürme fonksiyonu tanımla (boyutları ayarla, normalleştirme yap vb.)


# Veri kümesi ve veri yükleyici oluştur
input_folder = '.\\noises'
dataset = CustomDataset(input_folder)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Modeli ve kayıp fonksiyonunu tanımla
model = model.model
# model.load_state_dict(torch.load("input_diffusion_model.pth"))
# model.eval()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.004)
first_image = cv2.imread("1.jpeg")
first_image = cv2.resize(first_image,(28,28))
first_image = cv2.cvtColor(first_image,cv2.COLOR_BGR2GRAY)
first_image = torch.from_numpy(first_image)
# Eğitim döngüsü
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        # print(inputs[0].shape)
      

      
        # print(inputs.shape,"inputs")
        outputs = model(inputs)
        # image1 = inputs[0].detach().numpy()
        # image1 = cv2.resize(image1,(28,28))
        # cv2.imshow("input",image1)
        # image2 = targets[0].detach().numpy()
        # image2 = cv2.resize(image2,(28,28))
        # cv2.imshow("target",image2)
        # cv2.waitKey(0)
        # print(outputs.shape,targets.shape)
        # print("targets_outputs:",targets.shape,outputs.shape   )
        # print(outputs.shape)
        # outputs[0] = flatten_image(outputs[0])
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    output_image = model(first_image).detach().numpy() * 255
    output_image = reshape(output_image)
    cv2.imwrite(f"tests/test_{str(epoch)}.jpg",output_image)
    if(epoch % 5 == 0):
        torch.save(model.state_dict(), 'input_diffusion_model.pth')
