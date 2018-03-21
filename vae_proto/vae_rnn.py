import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Variable as Var

class VAEModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, lat_dim=33):
        super(VAEModel, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self.enc_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bidirectional=True, dropout=dropout)

        self.lat_dim = lat_dim
        self.fc_mu = nn.Linear(2 * nhid * nlayers * 2, lat_dim)  # 2 for bidirect, 2 for h and c
        self.fc_logvar = nn.Linear(2 * nhid * nlayers * 2, lat_dim)

        self.z_to_h = nn.Linear(lat_dim, nhid * nlayers)
        self.z_to_c = nn.Linear(lat_dim, nhid * nlayers)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)



        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def enc(self, input):
        """
        Encoding the input, output embedding, hidden (transformed from z), mu, logvar
        :param input: input sequence
        :return: embedding, hidden(from z), mu, logvar
        """
        batch_sz = input.size()[1]
        emb = self.drop(self.encoder(input))

        mu, logvar = self.encode(emb)
        z = self.reparameterize(mu, logvar)

        hidden = self.convert_z_to_hidden(z, batch_sz)
        return emb, hidden, mu, logvar

    def encode(self, emb):
        batch_sz = emb.size()[1]
        # self.enc_rnn.flatten_parameters()
        _, hidden = self.enc_rnn(emb)
        x = torch.cat(hidden, dim=0).permute(1, 0, 2).contiguous().view(batch_sz, -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def forward(self, input):

        batch_sz = input.size()[1]
        emb, hidden, mu, logvar = self.enc(input)

        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), mu, logvar

    def forward_decode(self, args, input, ntokens):
        seq_len = input.size()[0]
        batch_sz = input.size()[1]
        emb, hidden, mu, logvar = self.enc(input)
        # emb: seq_len, batchsz, hid_dim
        # hidden: ([2(nlayers),10(batchsz),200],[])

        outputs_prob = Var(torch.FloatTensor(seq_len, batch_sz, ntokens))
        if args.cuda:
            outputs_prob = outputs_prob.cuda()

        outputs = torch.LongTensor(seq_len, batch_sz)

        # First time step sos
        sos = Var(torch.ones(batch_sz).long())
        if args.cuda:
            sos = sos.cuda()
        emb_t = self.drop(self.encoder(sos))

        for t in range(seq_len):
            # input (seq_len, batch, input_size)
            # print(emb_t.size())
            emb_t = emb_t.unsqueeze(0)
            output, hidden = self.rnn(emb_t, hidden)
            output_prob = self.decoder(self.drop(output))
            # print(output_prob.size())
            output_prob = output_prob.squeeze(0)
            outputs_prob[t] = output_prob
            value, ind =torch.topk(output_prob,1,dim=1)
            # print(ind.size())
            emb_t = self.drop(self.encoder(ind.squeeze(1)))
            # print(emb_t.size())
            outputs[t] = ind.squeeze(1).data
            # exit()
        return outputs_prob, outputs

    def convert_z_to_hidden(self, z, batch_sz):
        h = self.z_to_h(z).view(batch_sz, 2, -1).permute(1, 0, 2).contiguous()
        c = self.z_to_c(z).view(batch_sz, 2, -1).permute(1, 0, 2).contiguous()
        return (h, c)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


def kld(mu, logvar, kl_weight=1):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    bsz = mu.size()[0]
    # print(bsz)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) /bsz
    KLD *= kl_weight
    return KLD
