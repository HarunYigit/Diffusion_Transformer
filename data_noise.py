import os
import cv2
import torch
from torchvision import transforms
import random
import numpy as np
# Gürültü ekleme fonksiyonu
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

# Giriş ve çıkış klasörleri
input_folder = "outputs"
output_folder = "noises"

# Eğer çıkış klasörü yoksa oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Gürültü parametreleri
noise_level = 0.01
ind = 0
noise_count = 75
def save_with_noise(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image,(28,28))
    first_image = image.copy()
    for iteration_count in range(1,noise_count):
        if "_noise_" in filename:
            iteration_count_str = str(iteration_count - 1)
            filename = filename.replace("_noise_" + iteration_count_str, "")
        image = add_percentage_noise(torch.tensor(image),  noise_percentage=iteration_count/ 1000)
        # cv2.imshow("imge",image)
        # cv2.waitKey(0)
        last = "." + filename.split(".")[-1]
        file_name_un_last = filename.replace(last,"")
        file_name_un_last += "_noise_" + str(noise_count - iteration_count)
        output_path = file_name_un_last + last
        output_path = output_path.replace(input_folder,output_folder)
        cv2.imwrite(output_path, image)
    iteration_count += 1
    image = add_percentage_noise(torch.tensor(image),  noise_percentage=75/ 1000)
    last = "." + filename.split(".")[-1]
    file_name_un_last = filename.replace(last,"")
    file_name_un_last += "_noise_" + str(73)
    output_path = file_name_un_last + last
    output_path = output_path.replace(input_folder,output_folder)
    cv2.imwrite(output_path.replace("noises\\","lastest_noises\\"), first_image)
    print(output_path)

# outputs klasöründeki dosyaları dolaş
ind = 0
for filename in os.listdir(input_folder):
    # Dosya yolu oluştur
    input_path = os.path.join(input_folder, filename)
    
    save_with_noise(input_path)
    ind += 1
    if ind > 50:
        break
