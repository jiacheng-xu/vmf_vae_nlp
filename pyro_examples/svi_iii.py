from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable
import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI
import sys


def param_abs_error(name, target):
    return torch.sum(torch.abs(target - pyro.param(name))).data.numpy()[0]


class BernoulliBetaExample(object):
    def __init__(self):
        # the two hyperparameters for the beta prior
        self.alpha0 = Variable(torch.Tensor([10.0]))
        self.beta0 = Variable(torch.Tensor([10.0]))
        # the dataset consists of six 1s and four 0s
        self.data = Variable(torch.zeros(10,1))
        self.data[0:6, 0].data = torch.ones(6)
        self.n_data = self.data.size(0)
        # compute the alpha parameter of the exact beta posterior
        self.alpha_n = self.alpha0 + self.data.sum()
        # compute the beta parameter of the exact beta posterior
        self.beta_n = self.beta0 - self.data.sum() + Variable(torch.Tensor([self.n_data]))
        # for convenience compute the logs
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)

    def setup(self):
        # initialize values of the two variational parameters
        # set to be quite close to the true values
        # so that the experiment doesn't take too long
        self.log_alpha_q_0 = Variable(torch.Tensor([np.log(15.0)]), requires_grad=True)
        self.log_beta_q_0 = Variable(torch.Tensor([np.log(15.0)]), requires_grad=True)

    def model(self, use_decaying_avg_baseline):
        # sample `latent_fairness` from the beta prior
        f = pyro.sample("latent_fairness", dist.beta, self.alpha0, self.beta0)
        # use iarange to indicate that the observations are
        # conditionally independent given f and get vectorization
        with pyro.iarange("data_iarange"):
            # observe all ten datapoints using the bernoulli likelihood
            pyro.observe("obs", dist.bernoulli, self.data, f)

    def guide(self, use_decaying_avg_baseline):
        # register the two variational parameters with pyro
        log_alpha_q = pyro.param("log_alpha_q", self.log_alpha_q_0)
        log_beta_q = pyro.param("log_beta_q", self.log_beta_q_0)
        alpha_q, beta_q = torch.exp(log_alpha_q), torch.exp(log_beta_q)
        # sample f from the beta variational distribution
        baseline_dict = {'use_decaying_avg_baseline': use_decaying_avg_baseline,
                         'baseline_beta': 0.90}
        # note that the baseline_dict specifies whether we're using
        # decaying average baselines or not
        pyro.sample("latent_fairness", dist.beta, alpha_q, beta_q,
                    baseline=baseline_dict)

    def do_inference(self, use_decaying_avg_baseline, tolerance=0.05):
        # clear the param store in case we're in a REPL
        pyro.clear_param_store()
        # initialize the variational parameters for this run
        self.setup()
        # setup the optimizer and the inference algorithm
        optimizer = optim.Adam({"lr": .0008, "betas": (0.93, 0.999)})
        svi = SVI(self.model, self.guide, optimizer, loss="ELBO", trace_graph=True)
        print("Doing inference with use_decaying_avg_baseline=%s" % use_decaying_avg_baseline)

        # do up to 10000 steps of inference
        for k in range(10000):
            svi.step(use_decaying_avg_baseline)
            if k % 100 == 0:
                print('.', end='')
                sys.stdout.flush()

            # compute the distance to the parameters of the true posterior
            alpha_error = param_abs_error("log_alpha_q", self.log_alpha_n)
            beta_error = param_abs_error("log_beta_q", self.log_beta_n)

            # stop inference early if we're close to the true posterior
            if alpha_error < tolerance and beta_error < tolerance:
                break

        print("\nDid %d steps of inference." % k)
        print(("Final absolute errors for the two variational parameters " +
               "(in log space) were %.4f & %.4f") % (alpha_error, beta_error))

# do the experiment
bbe = BernoulliBetaExample()
bbe.do_inference(use_decaying_avg_baseline=True)
bbe.do_inference(use_decaying_avg_baseline=False)