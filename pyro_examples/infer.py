from __future__ import print_function

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI
from pyro.optim import Adam
from torch.autograd import Variable

# clear the param store in case we're in a REPL
pyro.clear_param_store()

# create some data with 6 observed heads and 4 observed tails
data = []
for _ in range(6):
    data.append(Variable(torch.ones(1)))
for _ in range(4):
    data.append(Variable(torch.zeros(1)))


def model(data):
    # define the hyperparameters that control the beta prior
    alpha0 = Variable(torch.Tensor([10.0]))
    beta0 = Variable(torch.Tensor([10.0]))
    # sample f from the beta prior
    f = pyro.sample("latent_fairness", dist.beta, alpha0, beta0)
    # loop over the observed data
    for i in range(len(data)):
        # observe datapoint i using the bernoulli likelihood
        pyro.observe("obs_{}".format(i), dist.bernoulli, data[i], f)


def guide(data):
    # define the initial values of the two variational parameters
    # we initialize the guide near the model prior (except a bit sharper)
    log_alpha_q_0 = Variable(torch.Tensor([np.log(15.0)]), requires_grad=True)
    log_beta_q_0 = Variable(torch.Tensor([np.log(15.0)]), requires_grad=True)
    # register the two variational parameters with Pyro
    log_alpha_q = pyro.param("log_alpha_q", log_alpha_q_0)
    log_beta_q = pyro.param("log_beta_q", log_beta_q_0)
    alpha_q, beta_q = torch.exp(log_alpha_q), torch.exp(log_beta_q)
    # sample latent_fairness from Beta(alpha_q, beta_q)
    pyro.sample("latent_fairness", dist.beta, alpha_q, beta_q)


# setup the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss="ELBO", num_particles=7)

n_steps = 4000
# do gradient steps
for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print('.', end='')

# grab the learned variational parameters
alpha_q = torch.exp(pyro.param("log_alpha_q")).data.numpy()[0]
beta_q = torch.exp(pyro.param("log_beta_q")).data.numpy()[0]

# here we use some facts about the beta distribution
# compute the inferred mean of the coin's fairness
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * np.sqrt(factor)

print("\nbased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))
