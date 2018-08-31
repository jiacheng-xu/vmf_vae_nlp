import torch

from archive.nvdm.dist import Gauss, vMF, HighVarGauss


class BowVAE(torch.nn.Module):
    def __init__(self, vocab_size, n_hidden, n_lat, n_sample, batch_size, non_linearity, dist):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_lat = n_lat
        self.n_sample = n_sample
        self.non_linearity = non_linearity
        self.batch_size = batch_size
        self.dist_type = dist
        self.dropout = torch.nn.Dropout(p=0.2)
        # crit
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # Encoding
        self.enc_vec = torch.nn.Linear(self.vocab_size, self.n_hidden)
        self.active = torch.nn.LeakyReLU()
        self.enc_vec_2 = torch.nn.Linear(self.n_hidden, self.n_lat)

        if self.dist_type == 'nor':
            self.dist = Gauss(n_lat)
        elif self.dist_type == 'hnor':
            self.dist = HighVarGauss(n_lat)
        elif self.dist_type == 'vmf':
            self.dist = vMF(n_lat)
        else:
            raise NotImplementedError

        # Decoding
        self.out = torch.nn.Linear(self.n_lat, self.vocab_size)

    def forward(self, x, mask):
        batch_sz = x.size()[0]
        linear_x = self.enc_vec(x)
        active_x = self.active(linear_x)
        linear_x_2 = self.enc_vec_2(active_x)

        tup, kld, vecs = self.dist.build_bow_rep(linear_x_2, self.n_sample)

        ys = 0
        for i, v in enumerate(vecs):
            logit = torch.nn.functional.log_softmax(self.out(v))
            logit = self.dropout(logit)
            ys += torch.mul(x, logit)
        # out = self.out(vec)
        # logits = torch.nn.functional.log_softmax(out)
        # y = torch.mul(x, logits)

        y = ys / self.n_sample

        recon_loss = -torch.sum(y, dim=1, keepdim=False)

        kld = kld * mask
        recon_loss = recon_loss * mask

        total_loss = kld + recon_loss

        return recon_loss, kld, total_loss
