import numpy as np

import torch as T
import torch.nn as nn
import torch.optim as optim

def to_np(tensor):
    return tensor.detach().cpu().numpy()

def isin(ar, ar_collec):
    return (ar == ar_collec).all(axis=1).any()

def get_act(name):
    return {
        'relu': nn.ReLU(),
        'lrlu': nn.LeakyReLU(0.1),
        'silu': nn.SiLU(),
        'selu': nn.SELU(),
        'sigm': nn.Sigmoid(),
        'tanh': nn.Tanh(),
    }[name]

def get_loss(name, **kwargs):
    return {
        'l1loss': nn.L1Loss(reduction='none', **kwargs),
        'l2loss': nn.MSELoss(reduction='none', **kwargs),
        'hbloss': nn.HuberLoss(reduction='none', **kwargs),
        'celoss': nn.CrossEntropyLoss(reduction='none', **kwargs),
        'bcewll': nn.BCEWithLogitsLoss(reduction='none', **kwargs),
    }[name]

def get_opt(name, params, lr, **kwargs):
    if   name == 'adam': return optim.Adam(params, lr=lr, **kwargs)
    elif name == 'rmsp': return optim.RMSprop(params, lr=lr, **kwargs)
    elif name == 'sgd':  return optim.SGD(params, lr=lr, **kwargs)
    else:
        raise ValueError('No optimiser with name ', name)
