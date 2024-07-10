import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from typing import List

import utils
from articulate.rnn import RNN, RNNWithInit
from config import JointSet
 
    
class MVAE(nn.Module):
    ''' Replaced the gate network with the older version.
    '''
    
    def __init__(self):
        super().__init__()
        n_frame = 12 + 12 * JointSet.n_aug      # 240
        n_latent = 40   # !!!
        n_expert = 6
        n_gate_hidden = 64
        self.encode_fc1 = nn.Linear(n_frame * 2, n_frame * 2)
        self.encode_fc2 = nn.Linear(n_frame * 2, n_frame * 2)
        self.encode_fc3 = nn.Linear(n_frame * 2, n_frame * 2)
        self.encode_fc4 = nn.Linear(n_frame * 2, n_frame)
        self.encode_mu = nn.Linear(n_frame, n_latent)
        self.encode_logvar = nn.Linear(n_frame, n_latent)        
        # decoder
        self.decode_fc1 = nn.Linear(n_latent, n_frame)
        self.decode_fc2 = (nn.Parameter(torch.empty(n_expert, n_frame * 2, n_frame * 2)),
            nn.Parameter(torch.empty(n_expert, n_frame * 2)), F.elu)
        self.decode_out = (nn.Parameter(torch.empty(n_expert, n_frame * 2, n_frame)),
            nn.Parameter(torch.empty(n_expert, n_frame)), None)
        # initialize decode_fc2 and decode_out
        for idx, (weight, bias, _) in enumerate([self.decode_fc2, self.decode_out]):
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter(f'w{idx}', weight)
            self.register_parameter(f'b{idx}', bias)
        # gating network
        self.gate = nn.Sequential(
            nn.Linear(n_frame * 2, n_gate_hidden), nn.ELU(),
            nn.Linear(n_gate_hidden, n_gate_hidden), nn.ELU(),
            nn.Linear(n_gate_hidden, n_expert)
        )
        

    def _get_mixed_weight_and_bias(self, coeff, weight, bias):
        flat_weight = weight.flatten(start_dim=1, end_dim=2)
        mixed_weight = torch.matmul(coeff, flat_weight).view(
            coeff.shape[0], weight.shape[1], weight.shape[2])
        mixed_bias = torch.matmul(coeff, bias).unsqueeze(1)
        return mixed_weight, mixed_bias


    def encode(self, x, c):
        h_out = F.elu(self.encode_fc1(torch.cat([x, c], dim=1)))
        h_in = torch.cat([x, c], dim=1) + h_out
        h_out = F.elu(self.encode_fc2(h_in))
        h_out2 = F.elu(self.encode_fc3(h_out))
        h_out = F.elu(self.encode_fc4(h_out + h_out2))
        mu = self.encode_mu(h_out)
        logvar = self.encode_logvar(h_out)
        logvar = torch.clip(logvar, max=20)
        return mu, logvar
    
    
    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(sigma)
        return mu + epsilon * sigma
    
    
    def decode(self, z, c):
        z = F.elu(self.decode_fc1(z))
        coeff = F.softmax(self.gate(torch.cat([z, c], dim=1)), dim=1)
        # decode_fc2
        weight, bias, act = self.decode_fc2
        mixed_weight, mixed_bias = self._get_mixed_weight_and_bias(coeff, weight, bias)
        input_tensor = torch.cat([z, c], dim=1).unsqueeze(1)
        h = act(torch.baddbmm(mixed_bias, input_tensor, mixed_weight).squeeze(1))
        # decode out
        weight, bias, _ = self.decode_out
        mixed_weight, mixed_bias = self._get_mixed_weight_and_bias(coeff, weight, bias)
        input_tensor = (torch.cat([z, c], dim=1) + h).unsqueeze(1)
        out = torch.baddbmm(mixed_bias, input_tensor, mixed_weight).squeeze(1)
        return out
    
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z, c)
        return out, mu, logvar
     

class VAEDataset(torch.utils.data.Dataset):
    ''' Dataset for VAE.
    '''
    
    def __init__(self, data: List[torch.Tensor], label: List[torch.Tensor], split_size=-1):
        ''' Init a VAE dataset.
            Get the dataloader by torch.utils.data.DataLoader(dataset, **collate_fn=RNNDataset.collate_fn**)
        args:
            data: A list that contains sequences(tensors) in shape [num_frames, n_input].
            label: A list that contains sequences(tensors) in shape [num_frames, n_output].
            split_size: If positive, data and label will be split to list of small sequences.
        '''
        assert len(data) == len(label) and len(data) != 0
        if split_size > 0:
            self.data, self.label = [], []
            for td, tl in zip(data, label):
                self.data.extend(td.split(split_size))
                self.label.extend(tl.split(split_size))
        else:
            self.data = data
            self.label = label
        
    
    def __getitem__(self, idx):
        ''' Get item by its index.
        '''
        data = self.data[idx]
        label = self.label[idx]
        return data, label


    def __len__(self):
        return len(self.data)


    @staticmethod
    def collate_fn(x):
        ''' [[seq0, label0], [seq1, label1], [seq2, label2]] -> [[seq0, seq1, seq2], [label0, label1, label2]]
        '''
        return list(zip(*x))
    
    
class VAEWithInitDataset(VAEDataset):
    ''' The same as `VAEDataset`. Used for `VAEWithInit`.
    '''
    def __init__(self, data: List[torch.Tensor], label: List[torch.Tensor], split_size=-1, device=None):
        super(VAEWithInitDataset, self).__init__(data, label, split_size, device)

    def __getitem__(self, idx):
        ''' Get item by its index.
        '''
        data, label = super(VAEWithInitDataset, self).__getitem__(idx)
        return (data, label[0]), label
    

if __name__ == '__main__':
    model = MVAE()
    x = torch.randn(10, 240)
    c = torch.randn(10, 240)
    out, mu, logvar = model(x, c)
    print(out.shape, mu.shape, logvar.shape)
    