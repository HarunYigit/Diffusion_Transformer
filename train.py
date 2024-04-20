import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import json
import model
import utils
from data_loader import train_dataloader,test_dataloader
from torch.optim import lr_scheduler
from utils import reshape,flatten_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = json.load(open("config.json"))
epoch = config['epoch']
learning_rate = config['learning_rate']
batch_size = config['batch_size']
img_size = config['img_size']
diffusion_count = config['diffusion_count']
model = model.model.to(device)
# model.load_state_dict(torch.load("input_diffusion_model.pth"))
# model.eval()

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9,0.999))
# early_stopping = lr_scheduler.EarlyStopping(patience=5, verbose=True)

num_epochs = 100
data_len = len(train_dataloader)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    ind = 0
    batch_ind = 0
    for index,inputs in  enumerate(train_dataloader):
        if index == len(train_dataloader) - 2:
            break
        inputs = inputs.to(device)
        optimizer.zero_grad()
        targets = []
        batch_ind += 1
        for iteration_count in range(diffusion_count):
            optimizer.zero_grad()
            targets = []
            for input in inputs:
                input = input.view(-1)
                target = torch.from_numpy(utils.add_percentage_noise(torch.tensor(input).to('cpu'),  noise_percentage=iteration_count/ diffusion_count)).to(device)
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

        model.eval()
        test_inputs = utils.get_test_batch(batch_size)
        output_images = model(test_inputs).to("cpu").detach().numpy() * 255
        for reshape_index,i in enumerate(output_images):
            reshape_image = reshape(i)
            cv2.imwrite(f"tests_current/test_{reshape_index}.jpg",cv2.resize(reshape_image,(280,280)))
        cv2.imwrite(f"tests/test_{str(epoch)}.jpg",cv2.resize(reshape(output_images[0]),(280,280)))
        test_loss = 0
        with torch.no_grad():
            for test_index,test_inputs in enumerate(test_dataloader):
                if test_index == index:
                    test_inputs = test_inputs.to(device)
                    test_targets = []  # Test veri kümesi için hedefler
                    for test_input in test_inputs:
                        test_input = test_input.view(-1)
                        test_target = torch.from_numpy(utils.add_percentage_noise(torch.tensor(test_input).to('cpu'), noise_percentage=iteration_count / 20)).to(device)
                        test_target = test_target.view(-1)
                        test_target = test_target.to(torch.long)
                        test_targets.append(test_target)
                    test_targets = torch.stack(test_targets, dim=0)
                    test_output = model(test_targets)
                    test_input = test_input.to(torch.float) / 255.0
                    test_loss = criterion(test_output, test_input).item()
        print(f"Batch_{batch_ind}/{data_len} Loss={loss.item()} Epoch {epoch + 1}, Test Loss: {test_loss}")
        model.train()
        torch.save(model.state_dict(), 'input_diffusion_model.pth')

    ind += 1
    epoch_loss = running_loss / len(train_dataloader)
    print(f'\nEpoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')