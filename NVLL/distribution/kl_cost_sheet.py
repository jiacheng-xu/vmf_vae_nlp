from NVLL.distribution.vmf_only import vMF
x =[32,64,128,256,512]
import numpy as np
import torch
from scipy import special as sp


def _vmf_kld(k, d):
    tmp = (k * ((sp.iv(d / 2.0 + 1.0, k) + sp.iv(d / 2.0, k) * d / (2.0 * k)) / sp.iv(d / 2.0, k) - d / (2.0 * k)) \
           + d * np.log(k) / 2.0 - np.log(sp.iv(d / 2.0, k)) \
           - sp.loggamma(d / 2 + 1) - d * np.log(2) / 2).real

    return str(tmp)

tab = [[0 for _ in range(len(x))] for _ in range(len(x))]
for idx, lat_dim in enumerate(x):
    for jdx, kappa in enumerate(x):
        tab[idx][jdx] = _vmf_kld(kappa, lat_dim)

for idx in range(len(tab)):
    cont = tab[idx]
    print('\t'.join(cont))