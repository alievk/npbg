import torch
import torch.nn as nn


def norm(num_features, tp='bn', **kwargs):
    if tp == 'bn':
        return nn.BatchNorm2d(num_features, **kwargs)
    elif tp == 'in':
        return nn.InstanceNorm2d(num_features, **kwargs)
    elif tp == 'none':
        return Identity()


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
