import numpy as np
import scipy

import torch
from torch.autograd import Variable


class vmf():
    def __init__(self, norm_eps, norm_max, kappa):
        self.kappa = kappa
        self.norm_eps = norm_eps
        self.norm_max = norm_max
        self.normclip = torch.nn.Hardtanh(0, norm_max - norm_eps)

    def sample_vMF(self, mu):
        """vMF sampler in pytorch.

        http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python

        Args:
            mu (Tensor): of shape (batch_size, 2*word_dim)
            kappa (Float): controls dispersion. kappa of zero is no dispersion.
        """
        batch_size, id_dim = mu.size()
        result_list = []
        for i in range(batch_size):
            munorm = mu[i].norm(p=2).expand(id_dim)
            munoise = self.add_norm_noise(munorm, self.norm_eps)
            if float(mu[i].norm().data.cpu().numpy()) > 1e-10:
                # sample offset from center (on sphere) with spread kappa
                w = self._sample_weight(self.kappa, id_dim)
                wtorch = Variable(w * torch.ones(id_dim))

                # sample a point v on the unit sphere that's orthogonal to mu
                v = self._sample_orthonormal_to(mu[i] / munorm, id_dim)

                # compute new point
                scale_factr = torch.sqrt(Variable(torch.ones(id_dim)) - torch.pow(wtorch, 2))
                orth_term = v * scale_factr
                muscale = mu[i] * wtorch / munorm
                sampled_vec = (orth_term + muscale) * munoise
            else:
                rand_draw = Variable(torch.randn(id_dim))
                rand_draw = rand_draw / torch.norm(rand_draw, p=2).expand(id_dim)
                rand_norms = (torch.rand(1) * self.norm_eps).expand(id_dim)
                sampled_vec = rand_draw * Variable(rand_norms)  # mu[i]
            result_list.append(sampled_vec)

        return torch.stack(result_list, 0)

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = Variable(torch.randn(dim))
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)

    def add_norm_noise(self, munorm, eps):
        """
        KL loss is - log(maxvalue/eps)
        cut at maxvalue-eps, and add [0,eps] noise.
        """
        trand = torch.rand(1).expand(munorm.size()) * eps
        return (self.normclip(munorm) + Variable(trand))

    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1  # since S^{n-1}
        b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)  # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
        x = (1. - b) / (1. + b)
        c = kappa * x + dim * np.log(1 - x ** 2)  # dim * (kdiv *x + np.log(1-x**2))

        while True:
            z = np.random.beta(dim / 2., dim / 2.)  # concentrates towards 0.5 as d-> inf
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1. - x * w) - c >= np.log(
                    u):  # thresh is dim *(kdiv * (w-x) + log(1-x*w) -log(1-x**2))
                return w


"""
def draw_p_noise( batch_size, edit_dim):
    rand_draw = Variable(torch.randn(batch_size, edit_dim))
    rand_draw = rand_draw / torch.norm(rand_draw, p=2, dim=1).expand(batch_size, edit_dim)
    rand_norms = (torch.rand(batch_size, 1) * self.norm_max).expand(batch_size, edit_dim)
    return rand_draw * Variable(rand_norms)
"""
