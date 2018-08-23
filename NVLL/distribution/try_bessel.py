import torch
from scipy import special as sp
import numpy as np
from NVLL.util.util import GVar
from torch import optim
from NVLL.util.gpu_flag import device


class BesselIv(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, dim, kappa):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(dim, kappa)

        kappa_copy = kappa.clone()
        m = sp.iv(dim, kappa_copy)
        x = torch.tensor(m).to(device)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # print('called')
        dim, kappa = ctx.saved_tensors
        grad_input = grad_output.clone()
        # grad = grad_input * (bessel_ive(dim - 1, kappa) - bessel_ive(dim, kappa) * (dim + kappa) / kappa)
        grad = grad_input * (bessel_iv(dim - 1, kappa) + bessel_iv(dim + 1, kappa)) * 0.5
        return None, grad


bessel_iv = BesselIv.apply
nonneg = torch.nn.ReLU()
hid_dim = 100
d = torch.tensor(20.0).to(device)

inp = torch.ones(hid_dim).to(device)
func_kappa = torch.nn.Linear(hid_dim, 1)
func_kappa.to(device)

optimizer = optim.Adam(list(func_kappa.parameters()), lr=0.0001)
# kappa = torch.tensor(10.0,requires_grad=True).to(device)

# save = True
# res = torch.autograd.gradcheck(bessel_iv, (dim, kappa), raise_exception=True)
#
# print(res) # res should be True if the gradients are correct.


for i in range(100):
    optimizer.zero_grad()
    k = func_kappa(inp)
    k_non_neg = nonneg(k)
    # res = bessel_iv(dim, kappa)

    first = k * bessel_iv(d / 2, k) / bessel_iv(d / 2 - 1, k)
    second = (d / 2 - 1) * torch.log(k) - torch.log(bessel_iv(d / 2 - 1, k))
    combin = first + second
    print(combin)
    goal = torch.tensor(100.0).to(device)

    loss = torch.nn.MSELoss()(combin, goal)
    loss.backward()
    optimizer.step()
