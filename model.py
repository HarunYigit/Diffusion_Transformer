
import json
import torch
import numpy as np
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    def __init__(self,input_size, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5,batch_size: int = 10):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model,dropout,max_len=input_size).to(device)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout).to(device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers).to(device)
        self.embedding = nn.Embedding(ntoken, d_model).to(device)
        print("ntoken:",ntoken)
        self.d_model = d_model
        self.batch_size = batch_size
        self.ntoken = ntoken
        self.input_size = input_size
        self.linear = nn.Linear(input_size * d_model , input_size).to(device)
        self.relu = nn.LeakyReLU(0.01)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)



    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        """
        [seq_len, batch_size, embedding_dim]
        [4096,3,4]
        """
        src = src.permute(1, 0, 2)

        src = self.pos_encoder(src)


        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        """
        (seq, batch, feature)
        (4096,3)
        """
        output = self.transformer_encoder(src, src_mask)
        src = None
        output = output.permute(1, 0, 2)
        output  = torch.flatten(output, start_dim=1)  
        output = output.view(output.shape[0], -1)
        output = self.linear(output)
        return output



config = json.load(open("config.json"))
img_size = config['img_size']
batch_size = config['batch_size']
input_size =  img_size*img_size
n_token = 256
hidden_size = config['hidden_size']
num_layers =  config['num_layers']
num_heads = config['num_heads']
dropout =  config['dropout']
d_model = config['d_model']
model = DiffusionTransformer(input_size,n_token,d_model,num_heads,hidden_size,num_layers,dropout,batch_size).to(device)
total_params = 0
for param in model.parameters():
    total_params += param.numel()
print("total_param",total_params)
