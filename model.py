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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_heads: int, dropout: float = 0.1):
        super(DiffusionTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)

        # Embedding layer
        self.embedding = nn.Linear(input_size, hidden_size)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, input_size)

        self.init_weights()

    def init_weights(self):
        # Initialize weights and biases
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, src: Tensor, mask: Tensor = None) -> Tensor:
        # Apply embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.hidden_size)
        src = self.positional_encoding(src)

        # Apply transformer encoder
        output = self.transformer_encoder(src, mask)

        # Apply output layer
        output = self.output_layer(output)

        return output
    
img_size =  64*64*3
hidden_size = 2
num_layers =  2
num_heads = 2 
dropout =  0.2
model = DiffusionTransformer(img_size,hidden_size,num_layers,num_heads,dropout).to("cpu")