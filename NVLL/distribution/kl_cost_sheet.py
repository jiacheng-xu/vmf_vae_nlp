from NVLL.distribution.vmf_only import vMF

# x =[10, 25,50,100,200,400]
x = [50, 100, 150, 200, 250, 300, 350, 400]
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
color_bank = ['purple', 'magenta', 'pink', 'red', 'orange', 'lime', 'cyan', 'blue']
marker_bank = ['star', 'o', '|', 'triangle', 'x', '+', '-', 'square']

for idx in range(len(tab)):
    # lat dim
    s = "\\addplot [color={},mark={},]\ncoordinates {{".format(color_bank[idx], marker_bank[idx])
    cont = tab[idx]
    for j in range(len(tab)):
        s += "({},{})".format(x[j], cont[j])
    s += "};\n"
    print(s)
