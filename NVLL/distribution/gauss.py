import torch.nn as nn
import torch
import numpy as np
from NVLL.util.gpu_flag import device


class Gauss(nn.Module):
    # __slots__ = ['lat_dim', 'logvar', 'mean']

    def __init__(self, hid_dim, lat_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.lat_dim = lat_dim
        self.func_mean = torch.nn.Linear(hid_dim, lat_dim)
        self.func_logvar = torch.nn.Linear(hid_dim, lat_dim)
        # self.gate_mean = nn.Parameter(torch.rand(1))
        # self.gate_var = nn.Parameter(torch.rand(1))

    def estimate_param(self, latent_code):
        mean = self.func_mean(latent_code)
        logvar = self.func_logvar(latent_code)
        return {'mean': mean, 'logvar': logvar}

    def compute_KLD(self, tup):
        mean = tup['mean']
        logvar = tup['logvar']

        kld = -0.5 * torch.sum(1 - torch.mul(mean, mean) +
                               2 * logvar - torch.exp(2 * logvar), dim=1)
        return kld

    def sample_cell(self, batch_size):
        eps = torch.autograd.Variable(torch.normal(torch.zeros((batch_size, self.lat_dim))))
        eps.to(device)
        # if torch.cuda.is_available() and GPU_FLAG:
        #     eps = eps.cuda()
        return eps.unsqueeze(0)

    def build_bow_rep(self, lat_code, n_sample):
        batch_sz = lat_code.size()[0]
        tup = self.estimate_param(latent_code=lat_code)
        mean = tup['mean']
        logvar = tup['logvar']

        kld = self.compute_KLD(tup)
        if n_sample == 1:
            eps = self.sample_cell(batch_size=batch_sz)
            vec = torch.mul(torch.exp(logvar), eps) + mean
            return tup, kld, vec

        vecs = []
        for ns in range(n_sample):
            eps = self.sample_cell(batch_size=batch_sz)
            vec = torch.mul(torch.exp(logvar), eps) + mean
            vecs.append(vec)
        vecs = torch.cat(vecs, dim=0)
        return tup, kld, vecs

    def get_aux_loss_term(self, tup):

        return torch.from_numpy(np.zeros([1]))
