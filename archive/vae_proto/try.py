import torch
import numpy
import scipy.special

import numpy as np

mu, kappa = 2.0, 0  # mean and dispersion

# s = np.random.vonmises(mu, kappa, 1000)
# print(s)
#
# import matplotlib.pyplot as plt
# from scipy.special import i0
# plt.hist(s, 50, normed=True)
# x = np.linspace(-np.pi, np.pi, num=51)
# y = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa))
# plt.plot(x, y, linewidth=2, color='r')
# plt.show()

# x = scipy.special.iv(100, 0)
# a = 40
# y = 2 * numpy.power(scipy.pi,a/2+0.5 )
# x = scipy.special.gamma(a/2 + 0.5)
# print(y)
# print(x)
# print(y/x)


# x = scipy.special.iv(100, 0)
#
# k_1 = 30
# k_2 = 0.1
# dim=100
#
# def func_c(k, p):
#     # numer  = numpy.power(k,p/2-1)
#     # denom = scipy.special.iv(p/2-1, k)
#     x = numpy.power(k,p/2-1) / scipy.special.iv(p/2-1, k)
#     return x
# print(func_c(k_2, dim))
# print(func_c(k_1, dim) / func_c(k_2, dim))
#
#


with open('/Users/jcxu/Box/388NLP/hw3/penn-dependencybank/wsj-conllx/wsj_00.conllx', 'r') as f:
    lines = f.read().splitlines()
    cnt = 50
    buff = ''
    for l in lines:
        # print(l)
        if l == '':
            print('!')
            cnt -= 1
        if cnt == 0:
            print(buff)
            exit()
        else:
            buff = l
