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
from utils import generate_noise_image,flatten_image,reshape


model = model.model
model.load_state_dict(torch.load("input_diffusion_model.pth"))
model.eval()
first_image = generate_noise_image(64,64) * 255
first_image = first_image.astype(np.uint8)
print(first_image)
# cv2.imshow("random_image",cv2.resize(first_image,(280,280)))
# cv2.waitKey(0)
# first_image = cv2.cvtColor(first_image,cv2.COLOR_BGR2GRAY)
cv2.imwrite("test_noise.jpg",first_image)
first_image = flatten_image(first_image)
first_image = torch.from_numpy(first_image).to("cuda")
output_image = model(first_image)
output_image = output_image.to("cpu").detach().numpy() 
output_image = reshape(output_image)*255
for i in range(75):
    # print(first_image[0:5])
    first_image = model(first_image).to("cpu")
    first_image = first_image.detach().numpy() * 255
    first_image = first_image.astype(np.uint8)
    show_image = reshape(first_image)
    first_image = torch.from_numpy(first_image).to("cuda")
    # print("2:",first_image[0:5],show_image[0:5])
    show_image = cv2.resize(show_image,(280,280))
    # cv2.imshow("show_image",show_image.astype(np.uint8))
    # cv2.waitKey(0)
    # first_image = add_percentage_noise(first_image.detach().numpy(),0.01).astype(np.float64)
    # first_image = torch.from_numpy(first_image)
print(output_image[0][0:5])
first_image = first_image.to("cpu").detach().numpy()
first_image = reshape(first_image)
# print(output_image)
cv2.imwrite("test_75.jpg",first_image)
cv2.imwrite("test.jpg",output_image)
print("Resim kaydedildi.")