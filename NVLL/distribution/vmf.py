import numpy as np
import torch
from scipy import special as sp

from NVLL.util.util import GVar


class vMF(torch.nn.Module):
    def __init__(self, lat_dim, kappa=0):
        super().__init__()
        self.lat_dim = lat_dim
        self.func_mu = torch.nn.Linear(lat_dim, lat_dim)
        self.kappa = kappa
        self.norm_eps = 1
        self.normclip = torch.nn.Hardtanh(0, 10 - 1)

    def estimate_param(self, latent_code):
        mu = self.mu(latent_code)
        return {'mu': mu}

    def compute_KLD(self):
        kld = GVar(torch.zeros(1))
        return kld

    def vmf_unif_sampler(self, mu):
        batch_size, id_dim = mu.size()
        result_list = []
        for i in range(batch_size):
            munorm = mu[i].norm().expand(id_dim)
            munoise = self.add_norm_noise(munorm, self.norm_eps)
            if float(mu[i].norm().data.cpu().numpy()) > 1e-10:
                # sample offset from center (on sphere) with spread kappa
                w = self._sample_weight(self.kappa, id_dim)
                wtorch = GVar(w * torch.ones(id_dim))

                # sample a point v on the unit sphere that's orthogonal to mu
                v = self._sample_orthonormal_to(mu[i] / munorm, id_dim)

                # compute new point
                scale_factr = torch.sqrt(GVar(torch.ones(id_dim)) - torch.pow(wtorch, 2))
                orth_term = v * scale_factr
                muscale = mu[i] * wtorch / munorm
                sampled_vec = (orth_term + muscale) * munoise
            else:
                rand_draw = GVar(torch.randn(id_dim))
                rand_draw = rand_draw / torch.norm(rand_draw, p=2).expand(id_dim)
                rand_norms = (torch.rand(1) * self.norm_eps).expand(id_dim)
                sampled_vec = rand_draw * GVar(rand_norms)  # mu[i]
            result_list.append(sampled_vec)

        return torch.stack(result_list, 0)

    def vmf_sampler(self, mu):
        mu = mu.cpu()
        batch_size, id_dim = mu.size()
        result_list = []
        for i in range(batch_size):
            munorm = mu[i].norm().expand(id_dim)  # TODO norm p=?
            if float(mu[i].norm().data.cpu().numpy()) > 1e-10:
                # sample offset from center (on sphere) with spread kappa
                # w = self._sample_weight(self.kappa, id_dim)     # TODO mine?

                w = vMF.sample_vmf_w(self.kappa, id_dim)

                wtorch = GVar(w * torch.ones(id_dim))

                # sample a point v on the unit sphere that's orthogonal to mu
                v = self._sample_orthonormal_to(mu[i] / munorm, id_dim)
                # v= vMF.sample_vmf_v(mu[i])

                # compute new point
                scale_factr = torch.sqrt(GVar(torch.ones(id_dim)) - torch.pow(wtorch, 2))
                orth_term = v * scale_factr

                muscale = mu[i] * wtorch / munorm
                sampled_vec = (orth_term + muscale) * munorm
            else:
                rand_draw = GVar(torch.randn(id_dim))
                rand_draw = rand_draw / torch.norm(rand_draw, p=2).expand(id_dim)
                rand_norms = (torch.rand(1) * self.norm_eps).expand(id_dim)
                sampled_vec = rand_draw * GVar(rand_norms)  # mu[i]
            result_list.append(sampled_vec)

        return torch.stack(result_list, 0).cuda()

    def build_bow_rep(self, lat_code, n_sample):
        batch_sz = lat_code.size()[0]
        tup = self.estimate_param(latent_code=lat_code)

        kld = self.compute_KLD()

        vecs = []
        for ns in range(n_sample):
            vec = self.vmf_unif_sampler(tup['mu'])
            vecs.append(vec)

        # eps = self.vmf_sampler(tup['mu'])
        return tup, kld, vecs

    @staticmethod
    def _sample_weight(kappa, dim):
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

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = GVar(torch.randn(dim))
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)

    @staticmethod
    def sample_vmf_v(mu):
        import scipy.linalg as la
        mat = np.matrix(mu)

        if mat.shape[1] > mat.shape[0]:
            mat = mat.T

        U, _, _ = la.svd(mat)
        nu = np.matrix(np.random.randn(mat.shape[0])).T
        x = np.dot(U[:, 1:], nu[1:, :])
        return x / la.norm(x)

    @staticmethod
    def sample_vmf_w(kappa, m):

        b = (-2 * kappa + np.sqrt(4. * kappa ** 2 + (m - 1) ** 2)) / (m - 1)
        a = (m - 1 + 2 * kappa + np.sqrt(4 * kappa ** 2 + (m - 1) ** 2)) / 4
        d = 4 * a * b / (1 + b) - (m - 1) * np.log(m - 1)
        while True:
            z = np.random.beta(0.5 * (m - 1), 0.5 * (m - 1))
            W = (1 - (1 + b) * z) / (1 + (1 - b) * z)
            T = 2 * a * b / (1 + (1 - b) * z)
            u = np.random.uniform(0, 1)
            if (m - 1) * np.log(T) - T + d >= np.log(u):
                return W

    def add_norm_noise(self, munorm, eps):
        """
        KL loss is - log(maxvalue/eps)
        cut at maxvalue-eps, and add [0,eps] noise.
        """
        trand = torch.rand(1).expand(munorm.size()) * eps
        return (self.normclip(munorm) + GVar(trand))


def mod_Bessel_first_kind(kappa, d):
    # Modified Bessel function of the first kind
    return sp.iv(d, kappa)


def _vmfKL(k, d):
    return k * ((sp.iv(d / 2.0 + 1.0, k) \
                 + sp.iv(d / 2.0, k) * d / (2.0 * k)) / sp.iv(d / 2.0, k) - d / (2.0 * k)) \
           + d * np.log(k) / 2.0 - np.log(sp.iv(d / 2.0, k)) \
           - sp.loggamma(d / 2 + 1) - d * np.log(2) / 2


def _Constant(k, d):
    return np.power(k, d / 2 - 1) / (np.power(2 * np.pi, d / 2) * mod_Bessel_first_kind(d, k))


print(type(torch.ones(1) * _vmfKL(10, 200).real))
print(_Constant(0.1, 100))
