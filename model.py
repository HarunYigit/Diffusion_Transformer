import torch
import cv2
import numpy as np

def add_noise(image, noise_level=0.1, num_iterations=2):
    noisy_image = image.clone()  # Giriş görüntüsünü kopyala

    for _ in range(num_iterations):
        noise = torch.randn_like(image) * noise_level 
        noisy_image += noise  

    return noisy_image

image = cv2.imread('1.jpg')

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1))).float()

noisy_image = add_noise(image_tensor, noise_level=0.1, num_iterations=2)


import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DiffusionTransformer(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)



    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:

        src = self.embedding(src.to(torch.long)) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to("cpu")
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output
    
img_size =  64*64*3
hidden_size = 4
num_layers =  2
num_heads = 2 
dropout =  0.2
model = DiffusionTransformer(img_size,2,num_heads,hidden_size,num_layers,dropout).to("cpu")
