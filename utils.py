import torch
import cv2
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
# Görüntüyü OpenCV ile yükle
# image = cv2.imread('1.jpg')

# OpenCV görüntüsünü RGB formatında NumPy dizisine dönüştür
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# NumPy dizisini PyTorch tensorüne dönüştür
# image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1))).float()

# Örnek kullanım
# noisy_image = add_percentage_noise(image_tensor, noise_percentage=0.5)

# Gürültülü görüntüyü kontrol etmek için yazdır
# print(noisy_image)

# NumPy dizisini OpenCV kullanarak görsel olarak göster
# cv2.imshow('Image', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
# cv2.imshow('Noisy Image', cv2.cvtColor(noisy_image.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR))
# cv2.(0)
# cv2.destroyAllWindows()



