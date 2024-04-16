from matplotlib import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import model
import torchvision.transforms as transforms
import numpy as np
import random
def add_percentage_noise(image, noise_percentage=0.1):
    """
    image: Gürültü eklenecek olan giriş görüntüsü (numpy array)
    noise_percentage: Gürültünün oranı (0 ile 1 arasında bir sayı)
    """
    noisy_image = np.copy(image)   # Giriş görüntüsünün bir kopyasını oluştur
    last_shape = noisy_image.shape
    noisy_image = noisy_image.flatten()
    for i in range(len(noisy_image)):
        if random.random() < noise_percentage:
            noisy_image[i] = random.randint(0,255)
    noisy_image = np.reshape(noisy_image,last_shape) 
    
    return noisy_image

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
def generate_noise_image(height, width):
    # Rastgele gürültü matrisi oluşturma (0 ile 1 arasında değerler)
    noise_matrix = np.random.rand(28,28).astype(np.float64)
    return noise_matrix
model = model.model
model.load_state_dict(torch.load("input_diffusion_model.pth"))
model.eval()
first_image = generate_noise_image(28,28) * 255
# first_image = cv2.cvtColor(first_image,cv2.COLOR_BGR2GRAY)
cv2.imwrite("test_noise.jpg",first_image)
first_image = flatten_image(first_image)
first_image = torch.from_numpy(first_image)
output_image = model(first_image)
output_image = output_image.detach().numpy() 
output_image = reshape(output_image)*255
for i in range(5):
    print(first_image[0:5])
    first_image = model(first_image)
    print("2:",first_image[0:5])
    first_image = first_image.detach().numpy() * 255
    first_image = first_image.astype(np.uint8)
    first_image = torch.from_numpy(first_image)
    # first_image = add_percentage_noise(first_image.detach().numpy(),0.01).astype(np.float32)
    # first_image = torch.from_numpy(first_image)
print(output_image[0][0:5])
first_image = first_image.detach().numpy()
first_image = reshape(first_image)
# print(output_image)
cv2.imwrite("test_75.jpg",first_image)
cv2.imwrite("test.jpg",output_image)
print("Resim kaydedildi.")