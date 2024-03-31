import torch
import cv2
import numpy as np

def add_noise(image, noise_level=0.1, num_iterations=2):
    noisy_image = image.clone()  # Giriş görüntüsünü kopyala

    for _ in range(num_iterations):
        noise = torch.randn_like(image) * noise_level  # Gürültü tensorü oluştur
        noisy_image += noise  # Gürültüyü görüntüye ekle

    return noisy_image

# Görüntüyü OpenCV ile yükle
image = cv2.imread('1.jpg')

# OpenCV görüntüsünü RGB formatında NumPy dizisine dönüştür
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# NumPy dizisini PyTorch tensorüne dönüştür
image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1))).float()

# Örnek kullanım
noisy_image = add_noise(image_tensor, noise_level=0.1, num_iterations=2)

# Gürültülü görüntüyü kontrol etmek için yazdır
print(noisy_image)

# NumPy dizisini OpenCV kullanarak görsel olarak göster
cv2.imshow('Image', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
cv2.imshow('Noisy Image', cv2.cvtColor(noisy_image.numpy().transpose((1, 2, 0)), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
