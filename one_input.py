import model
from utils import generate_noise_image,flatten_image,reshape
import torch
import numpy as np
from data_loader import dataloader

mymodel = model.model
mymodel.eval()

for inputs in dataloader:
    mymodel(inputs.to(torch.long).to("cuda"))
    break