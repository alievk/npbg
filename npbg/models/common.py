import torch
import torch.nn as nn


def get_norm_layer(name='bn', **kwargs):
    if name == 'bn':
        return nn.BatchNorm2d
    elif name == 'in':
        return nn.InstanceNorm2d
    elif name == 'none':
        return Identity
    else:
        raise ValueError(f'{norm_layer} not supported')


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
