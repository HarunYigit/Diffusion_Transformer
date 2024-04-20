import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import os
import json
import model
import utils
import numpy as np
from data_loader import dataloader
from utils import reshape,flatten_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = json.load(open("config.json"))
epoch = config['epoch']
learning_rate = config['learning_rate']
batch_size = config['batch_size']
img_size = config['img_size']
model = model.model.to(device)
# model.load_state_dict(torch.load("input_diffusion_model.pth"))
# model.eval()
def get_test_batch(batch_size):
    noises = []
    for i in range(batch_size):
        first_image = utils.generate_noise_image(img_size,img_size) * 255
        first_image = flatten_image(first_image)
      #  first_image = torch.from_numpy(first_image).to(device).to(torch.long)
        noises.append(first_image)
    return torch.tensor(noises).to(device).to(torch.long)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9,0.999))

num_epochs = 100
data_len = len(dataloader)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    ind = 0
    batch_ind = 0
    for index,inputs in  enumerate(dataloader):
        if index == len(dataloader) - 2:
            break
        inputs = inputs.to(device)
        optimizer.zero_grad()
        targets = []
        batch_ind += 1
        for iteration_count in range(20):
            optimizer.zero_grad()
            targets = []
            for input in inputs:
                input = input.view(-1)
                target = torch.from_numpy(utils.add_percentage_noise(torch.tensor(input).to('cpu'),  noise_percentage=iteration_count/ 20)).to(device)
                target = target.view(-1)
                target = target.to(torch.long)
                targets.append(target)
            
            targets = torch.stack(targets, dim=0)
            output = model(targets)
            input = input.to(torch.float) / 255.0
            loss = criterion(output, input)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            inputs = targets
        print(f"Batch_{batch_ind}/{data_len} Loss={loss.item()}")
        if batch_ind % 3 == 0:
            model.eval()
            test_inputs = get_test_batch(batch_size)
            output_images = model(test_inputs).to("cpu").detach().numpy() * 255
            for reshape_index,i in enumerate(output_images):
                reshape_image = reshape(i)
                cv2.imwrite(f"tests_current/test_{reshape_index}.jpg",cv2.resize(reshape_image,(280,280)))
            cv2.imwrite(f"tests/test_{str(epoch)}.jpg",cv2.resize(reshape(output_images[0]),(280,280)))
            model.train()

    ind += 1
    epoch_loss = running_loss / len(dataloader)
    print(f'\nEpoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    torch.save(model.state_dict(), 'input_diffusion_model.pth')