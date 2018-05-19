from NVLL.distribution.vmf_batch import vMF
import numpy
import torch
from NVLL.util.util import GVar
x = [10	,25	,50	,100	,200	,400]

x = list(range(50,500,50))

x = list(range(10,160,15))
x = [100,200,300,400]
x = list(range(50,450,50))
print(x)

tab = [[] for _ in range(len(x))]
for idx, lat_dim in enumerate(x):
    for jdx, kappa in enumerate(x):
        tmp = vMF(lat_dim,lat_dim,kappa)


        vec = numpy.random.uniform(-1,1,lat_dim)
        vec = torch.FloatTensor(vec)
        vec = vec / torch.norm(vec)
        mu = GVar(vec).unsqueeze(0).expand(100,lat_dim)

        sampled = tmp.sample_cell(mu, None, kappa)
        sampled = sampled.squeeze()
        sim = torch.nn.functional.cosine_similarity(sampled, mu)
        tab[idx].append(tmp.kld.data[0])
for t in tab:
    t = [str(x) for x in t]
    print("\t".join(t))
print(tab)

