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
import json
from utils import generate_noise_image,flatten_image,reshape,get_test_batch
config = json.load(open("config.json"))
batch_size = config['batch_size']
img_size = config['img_size']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.model
model.load_state_dict(torch.load("input_diffusion_model.pth"))
model = model.to('cpu')
model.eval()
# test_inputs = get_test_batch(batch_size).to('cpu')
   
# test_inputs = model(test_inputs) * 255
# test_inputs = test_inputs.to(torch.long)
# for reshape_index,i in enumerate(test_inputs):

#     reshape_image = reshape(i)
#     cv2.imwrite(f"test_manuel/test_{reshape_index}.jpg",cv2.resize(reshape_image,(280,280)))

# test_inputs =[]
# for i in os.listdir("test_manuel"):
#     img = cv2.imread("test_manuel/" + i)
#     img = cv2.resize(img,(img_size,img_size))
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     img = flatten_image(img)
#     test_inputs.append(img)
# test_inputs= torch.tensor(test_inputs).to(torch.long)
# test_inputs = model(test_inputs) * 255
# test_inputs= test_inputs.to(torch.long)
# for reshape_index,i in enumerate(test_inputs):
#      reshape_image = reshape(i)
#      print(reshape_index,"kaydedildi")
#      cv2.imwrite(f"test_manuel/test_{reshape_index}.jpg",cv2.resize(reshape_image,(img_size,img_size)))