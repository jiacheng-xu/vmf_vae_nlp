import torch
from NVLL.util.util import GVar
from NVLL.distribution.gauss import Gauss
from NVLL.distribution.stable_vmf import vMF
from NVLL.distribution.unifvmf import unif_vMF

import numpy as np

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
            self.dist = unif_vMF(n_hidden, n_lat, kappa=self.args.kappa)
        else:
            raise NotImplementedError

        # Decoding
        self.out = torch.nn.Linear(self.n_lat, self.vocab_size)

    def forward(self, x):
        batch_sz = x.size()[0]

        linear_x = self.enc_vec(x)
        linear_x = self.dropout(linear_x)
        active_x = self.active(linear_x)
        linear_x_2 = self.enc_vec_2(active_x)

        tup, kld, vecs = self.dist.build_bow_rep(linear_x_2, self.n_sample)

        ys = 0
        for i, v in enumerate(vecs):
            logit = self.dropout(self.out(v))
            logit = torch.nn.functional.log_softmax(logit)
            ys += torch.mul(x, logit)
        # out = self.out(vec)
        # logits = torch.nn.functional.log_softmax(out)
        # y = torch.mul(x, logits)

        y = ys / self.n_sample

        recon_loss = -torch.sum(y, dim=1, keepdim=False)

        aux_loss = self.dist.get_aux_loss_term(tup)

        total_loss = kld + recon_loss + aux_loss

        return recon_loss, kld, aux_loss, total_loss, tup, vecs
