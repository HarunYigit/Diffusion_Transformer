import json
import os
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
    image =np.zeros((img_size,img_size))
    x = 0
    y = 0
    for i in flatten:
        image[y,x] = i
        y += 1  # Y indeksi artırılıyor
        if y == img_size:  # Y indeksi 28 olduğunda (satır bittiğinde), X indeksini artır ve Y'yi sıfırla
            x += 1
            y = 0
    return np.array(image)

def generate_noise_image(height, width):
    # Rastgele gürültü matrisi oluşturma (0 ile 1 arasında değerler)
    noise_matrix = np.random.rand(img_size,img_size).astype(np.float64)
    return noise_matrix
def preprocess_image(image_path, target_size=-1):
    if target_size == -1:
        target_size = (img_size,img_size)
    # Resmi oku
    input_image = cv2.imread(image_path)
    # Hedef boyuta yeniden boyutlandır
    input_image = cv2.resize(input_image, target_size)
    # Gri tonlamaya dönüştür
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    return input_image
def split_image_into_patches(image, patch_size=(4, 4)):
    patches = []
    height, width = image.shape
    patch_height, patch_width = patch_size
    # Resmi parçalara ayır
    for y in range(0, height, patch_height):
        for x in range(0, width, patch_width):
            patch = image[y:y+patch_height, x:x+patch_width]
            patches.append(patch)
    return patches

config = json.load(open("config.json"))
epoch = config['epoch']
img_size = config['img_size']