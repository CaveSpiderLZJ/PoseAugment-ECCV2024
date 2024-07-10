__all__ = ['RNNDataset']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from typing import List
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, n_input:int, n_output:int, n_hidden:int, n_rnn_layer:int=2,
            bidirectional:bool=True, rnn_dropout:float=0.0, linear_dropout:float=0.2, act=F.relu):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, dropout=rnn_dropout)
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = nn.Dropout(linear_dropout)
        self.act = act

    def forward(self, x, h=None):
        ''' Forward.
        args:
            x: a list of tensors.
        '''
        length = [item.shape[0] for item in x]
        x = pad_sequence(x)
        x = self.act(self.linear1(x))
        x = self.dropout(x)
        x = pack_padded_sequence(x, length, enforce_sorted=False)
        x, h = self.rnn(x, h)
        x = pad_packed_sequence(x)[0]
        x = self.linear2(x)
        return [x[:l, i].clone() for i, l in enumerate(length)], h
    
    
class RNNWithInit(RNN):
    ''' RNN with the initial hidden states regressed from the first output.
    '''
    
    def __init__(self, n_input:int, n_output:int, n_hidden:int, n_rnn_layer:int=2, bidirectional=True,
            rnn_dropout:float=0.0, linear_dropout:float=0.2, act=F.relu, init_is_output=True):
        r"""
        Init an RNNWithInit net.

        :param input_size: Input size.
        :param output_size: Output size.
        :param hidden_size: Hidden size for RNN.
        :param num_rnn_layer: Number of RNN layers.
        :param bidirectional: Whether if the RNN is bidirectional.
        :param dropout: Dropout after the input linear layer and in the rnn.
        :param load_weight_file: If not None and exists, weights will be loaded.
        """
        super().__init__(n_input, n_output, n_hidden, n_rnn_layer, bidirectional, rnn_dropout, linear_dropout, act)
        self.init_net = torch.nn.Sequential(
            torch.nn.Linear(n_output if init_is_output else n_input, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden * n_rnn_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden * n_rnn_layer, 2 * (2 if bidirectional else 1) * n_rnn_layer * n_hidden)
        )


    def forward(self, x, _=None):
        ''' Forward.
        args:
            x: A list of tensor like (x, x_init).
            _: Not used.
        returns:
            A list in length [batch_size] which contains tensors in shape [num_frames, output_size].
        '''
        
        x, x_init = list(zip(*x))
        nd, nh = self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), self.rnn.hidden_size
        h, c = self.init_net(torch.stack(x_init)).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
        return super(RNNWithInit, self).forward(x, (h, c))


class RNNDataset(torch.utils.data.Dataset):
    ''' Dataset for `net.RNN`.
    '''
    
    def __init__(self, data: List[torch.Tensor], label: List[torch.Tensor], split_size=-1, device=None):
        ''' Init an RNN dataset.

        Notes
        -----
        Get the dataloader by torch.utils.data.DataLoader(dataset, **collate_fn=RNNDataset.collate_fn**)

        If `split_size` is positive, `data` and `label` will be split to lists of small sequences whose lengths
        are not larger than `split_size`. Otherwise, it has no effects.

        If `augment_fn` is not None, `data` item will be augmented like `augment_fn(data[i])` in `__getitem__`.
        Otherwise, it has no effects.
        Probably not used in this project.

        Args
        -----
        :param data: A list that contains sequences(tensors) in shape [num_frames, input_size].
        :param label: A list that contains sequences(tensors) in shape [num_frames, output_size].
        :param split_size: If positive, data and label will be split to list of small sequences.
        :param device: The loaded data is finally copied to the device. If None, the device of data[0] is used.
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
        self.device = device if device is not None else data[0].device


    def __getitem__(self, idx):
        ''' Get item by its index.
        TODO: to(device) could be optimized.
        '''
        data = self.data[idx]
        label = self.label[idx]
        return data.to(self.device), label.to(self.device)


    def __len__(self):
        return len(self.data)


    @staticmethod
    def collate_fn(x):
        ''' [[seq0, label0], [seq1, label1], [seq2, label2]] -> [[seq0, seq1, seq2], [label0, label1, label2]]
        '''
        return list(zip(*x))
    

class RNNWithInitDataset(RNNDataset):
    r"""
    The same as `RNNDataset`. Used for `RNNWithInit`.
    """
    def __init__(self, data: List[torch.Tensor], label: List[torch.Tensor], split_size=-1, device=None):
        super(RNNWithInitDataset, self).__init__(data, label, split_size, device)

    def __getitem__(self, i):
        ''' Return extra label[0] compared with RNNDataset.
        '''
        data, label = super(RNNWithInitDataset, self).__getitem__(i)
        return (data, label[0]), label
    

class RNNLossWrapper:
    r"""
    Loss wrapper for `articulate.utils.torch.RNN`.
    First concat data, then compute loss.
    """
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, y_pred, y_true):
        return self.loss_fn(torch.cat(y_pred), torch.cat(y_true))
    
    
if __name__ == '__main__':
    x = [torch.randn(90, 90), torch.randn(100, 90), torch.randn(110, 90)]
    model = RNN(n_input=90, n_output=90, n_hidden=256, bidirectional=True, act=F.selu)
    out, h = model(x[:3])
    print(h[0].shape, h[1].shape)
