import time

import torch

from NVLL.util.util import GVar

from NVLL.util.gpu_flag import device

print(device)
#
# start = time.time()
# hard = torch.nn.Hardtanh()
# softmax = torch.nn.Softmax()
# for i in range(100):
#     x = torch.zeros(100000).cuda()
#     y = torch.rand(100000).cuda()
#     z = y * y * y
#     c = y * y / (y + y)
#     d = c * c + c
#     m = y + z + y
#     m = GVar(m)
#
#     for j in range(1000):
#         k = hard(m)
#         e = softmax(m + m)
#         q = softmax(m)
#
# print(time.time() - start)
import numpy as np


def _sample_weight(kappa, dim):
    """Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    dim = dim - 1  # since S^{n-1}
    b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)  # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
    x = (1. - b) / (1. + b)
    c = kappa * x + dim * np.log(1 - x ** 2)  # dim * (kdiv *x + np.log(1-x**2))
    cnt = 0
    while True:
        cnt += 1
        z = np.random.beta(dim / 2., dim / 2.)  # concentrates towards 0.5 as d-> inf
        w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
        u = np.random.uniform(low=0, high=1)
        if kappa * w + dim * np.log(1. - x * w) - c >= np.log(
                u):  # thresh is dim *(kdiv * (w-x) + log(1-x*w) -log(1-x**2))
            return w, cnt


kappa = [32, 64, 128]

# for k in kappa:
#     for d in kappa:
#         l = []
#         for _ in range(1000):
#             _, cnt =_sample_weight(k,d)
#             l.append(cnt)
#         print("{}\t{}\t{}".format(k,d,sum(l)/len(l)))
input = torch.FloatTensor()
torch.multinomial(input, 1, replacement=False)
