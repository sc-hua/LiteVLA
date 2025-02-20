import torch.nn as nn
import torch.nn.functional as F


def get_act(name):
    if name == 'relu':
        return nn.ReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'silu':
        return nn.SiLU()
    if name == 'elu1':
        return ELU_1()
    raise NotImplementedError(f'Unknown act: {name}')


class ELU_1(nn.Module):
    """ for linear attention """
    def forward(self, x):
        return F.elu(x) + 1
