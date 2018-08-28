from vmf_hypvae import *
import torch
import numpy as np


def kl_histogram_vs_uniform(samples):
    # uniform: 1/max_range
    max_range = 2 * np.pi
    num_splits = 1000
    unif = 1.0 / num_splits
    total_num = 0
    total_kl = 0.0
    for i in range(num_splits):
        lb = max_range * i / float(num_splits)
        ub = max_range * (i + 1) / float(num_splits)
        num_in_this = len(list(filter(lambda x: lb <= x and x < ub, samples)))
        total_num += num_in_this
        pi = float(num_in_this) / float(len(samples))
        if pi > 0.0:
            total_kl += pi * (np.log(pi) - np.log(unif))
    if total_num != len(samples):
        print("Unaccounted for! %d found" % total_num)
    print("Empirical KL: %f" % total_kl)


def check_kappa(kappa):
    print("Checking dim = 2, kappa = %f" % kappa)
    vmf_diff = VmfDiff(100, 100)
    dim = 2
    print("KL Guu %f" % KL_guu(kappa, dim))
    print("KL Davidson %f" % KL_davidson(kappa, dim))
    samples = []
    for i in range(0, 10000):
        # result = vmf_diff.sample_cell(torch.tensor([[1.0, 0.0]]), norm=0.0, kappa=torch.tensor([100.0]))
        result = vmf_diff.sample_cell(torch.tensor([[0.0, 1.0]]), norm=0.0, kappa=torch.tensor([kappa]))

        # print(result)
        x = result.data[0][0][0]
        y = result.data[0][0][1]
        if x > 0 and y > 0:
            angle_in_rads = np.arctan(y / x)
        elif x < 0 and y > 0:  # quadrant 2
            angle_in_rads = np.pi - np.arctan(-y / x)
        elif x < 0 and y < 0:  # quadrant 3
            angle_in_rads = np.pi + np.arctan(y / x)  # -y/-x
        elif x > 0 and y < 0:
            angle_in_rads = 2 * np.pi - np.arctan(-y / x)  # -y/-x
        # print(angle_in_rads)
        samples.append(angle_in_rads.item())
    kl_histogram_vs_uniform(samples)


if __name__ == "__main__":
    for kappa in [5.0, 10.0, 20.0, 30.0]:
        check_kappa(kappa)
