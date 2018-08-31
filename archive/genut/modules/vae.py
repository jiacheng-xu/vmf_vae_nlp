from torch.autograd.variable import Variable
import torch
from torch import nn

fudge = 1e-7
num_simulation = 5


class VI(nn.Module):
    def __init__(self):
        pass


def Q(hid_rep):
    z_mu = h @ Whz_mu + bhz_mu.repeat(h.size(0), 1)
    z_var = h @ Whz_var + bhz_var.repeat(h.size(0), 1)
    return z_mu, z_var


def sample_z_gauss(batch_sz, lat_dim, mu, log_var):
    eps = Variable(torch.randn(batch_sz, lat_dim))
    return mu + torch.exp(log_var / 2) * eps
