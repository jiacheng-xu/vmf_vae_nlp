from NVLL.distribution.vmf_batch import vMF
import numpy
import torch
from NVLL.util.util import GVar

x = [10, 25, 50, 100, 200, 400]

x = list(range(50, 450, 50))
x = [25, 100]
print(x)
k = list(range(5, 275, 20))
print(k)

tab_kl = [[] for _ in range(len(x))]
tab_sim = [[] for _ in range(len(x))]
for idx, lat_dim in enumerate(x):

    vec = numpy.random.uniform(-10, 10, lat_dim)
    vec = torch.FloatTensor(vec)
    vec = vec / torch.norm(vec)

    for jdx, kappa in enumerate(k):
        tmp = vMF(lat_dim, lat_dim, kappa)

        mu = GVar(vec).unsqueeze(0).expand(10000, lat_dim)

        sampled = tmp.sample_cell(mu, None, kappa)
        sampled = sampled.squeeze()
        sim = torch.nn.functional.cosine_similarity(sampled, mu)

        tab_kl[idx].append(tmp.kld.item())
        tab_sim[idx].append(torch.mean(sim).item())
# for t in tab:
#     t = [str(x) for x in t]
#     print("\t".join(t))
print("kld = {}".format(tab_kl))
print("cos = {}".format(tab_sim))
