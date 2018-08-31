import random

random.seed(2018)
import torch

from NVLL.distribution.gauss import Gauss
from NVLL.distribution.vmf_batch import vMF
from NVLL.distribution.vmf_unif import unif_vMF
from NVLL.distribution.vmf_hypvae import VmfDiff
from NVLL.util.util import GVar


class BowVAE(torch.nn.Module):
    def __init__(self, args, vocab_size, n_hidden, n_lat, n_sample, dist):
        super(BowVAE, self).__init__()
        self.args = args

        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_lat = n_lat
        self.n_sample = n_sample
        self.dist_type = dist
        self.dropout = torch.nn.Dropout(p=args.dropout)
        # crit
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # Encoding
        self.enc_vec = torch.nn.Linear(self.vocab_size, self.n_hidden)
        self.active = torch.nn.Tanh()
        self.enc_vec_2 = torch.nn.Linear(self.n_hidden, self.n_hidden)

        if self.dist_type == 'nor':
            self.dist = Gauss(n_hidden, n_lat)
        elif self.dist_type == 'vmf':
            self.dist = vMF(n_hidden, n_lat, kappa=self.args.kappa)
        elif self.dist_type == 'unifvmf':
            self.dist = unif_vMF(n_hidden, n_lat, kappa=self.args.kappa, norm_func=self.args.norm_func)
        elif self.dist_type == 'sph':
            self.dist = VmfDiff(n_hidden, n_lat)
        else:
            raise NotImplementedError

        # Decoding
        self.dec_linear = torch.nn.Linear(self.n_lat, self.n_hidden)
        self.dec_act = torch.nn.Tanh()

        self.out = torch.nn.Linear(self.n_hidden, self.vocab_size)

    def forward(self, x):
        batch_sz = x.size()[0]

        linear_x = self.enc_vec(x)
        linear_x = self.dropout(linear_x)
        active_x = self.active(linear_x)
        linear_x_2 = self.enc_vec_2(active_x)

        tup, kld, vecs = self.dist.build_bow_rep(linear_x_2, self.n_sample)
        # vecs: n_samples, batch_sz, lat_dim

        if 'redundant_norm' in tup:
            aux_loss = tup['redundant_norm'].view(batch_sz)
        else:
            aux_loss = GVar(torch.zeros(batch_sz))

        # stat
        avg_cos = BowVAE.check_dispersion(vecs)
        avg_norm = torch.mean(tup['norm'])
        tup['avg_cos'] = avg_cos
        tup['avg_norm'] = avg_norm

        flatten_vecs = vecs.view(self.n_sample * batch_sz, self.n_lat)
        flatten_vecs = self.dec_act(self.dec_linear(flatten_vecs))
        logit = self.dropout(self.out(flatten_vecs))
        logit = torch.nn.functional.log_softmax(logit, dim=1)
        logit = logit.view(self.n_sample, batch_sz, self.vocab_size)
        flatten_x = x.unsqueeze(0).expand(self.n_sample, batch_sz, self.vocab_size)
        error = torch.mul(flatten_x, logit)
        error = torch.mean(error, dim=0)

        recon_loss = -torch.sum(error, dim=1, keepdim=False)

        return recon_loss, kld, aux_loss, tup, vecs

    @staticmethod
    def cos(a, b):
        return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

    @staticmethod
    def check_dispersion(vecs):
        # vecs: n_samples, batch_sz, lat_dim
        num_sam = 10
        cos_sim = 0
        for i in range(num_sam):
            idx1 = random.randint(0, vecs.size(1) - 1)
            while True:
                idx2 = random.randint(0, vecs.size(1) - 1)
                if idx1 != idx2:
                    break
            cos_sim += BowVAE.cos(vecs[0][idx1], vecs[0][idx2])
        return cos_sim / num_sam
        # print("Avg cosine sim of mus across batch: " + repr(cos_sim / num_sam))
