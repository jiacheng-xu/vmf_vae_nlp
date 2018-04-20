import torch
import torch.nn as nn
import numpy
from NVLL.distribution.gauss import Gauss
from NVLL.distribution.vmf_only import vMF
from NVLL.util.util import GVar


class RNNVAE(nn.Module):
    """Container module with an optional encoder, a prob latent module, and a RNN decoder."""


    def __init__(self, args, enc_type, ntoken, ninp, nhid, lat_dim, nlayers, dropout=0.5, tie_weights=False):

        super(RNNVAE, self).__init__()
        self.FLAG_train = True
        self.args = args
        self.lat_dim = lat_dim
        self.nhid = nhid
        self.nlayers = nlayers  # layers for decoding part
        self.ninp = ninp
        self.ntoken = ntoken
        self.dist_type = args.dist  # Gauss or vMF = normal vae;
        # zero = no encoder and VAE, use 0 as start of decoding; sph = encode word embedding as bow and project to a unit sphere

        self.run = self.forward_fly if args.fly else self.forward_ground

        # VAE shared param
        self.drop = nn.Dropout(dropout)  # Need explicit dropout
        self.emb = nn.Embedding(ntoken, ninp)

        # VAE decoding part
        self.decoder_out = nn.Linear(nhid, ntoken)

        # VAE recognition part
        if self.dist_type == 'nor' or 'vmf' or 'sph':
            if enc_type == 'lstm':
                self.enc_lstm = nn.LSTM(ninp, nhid, 1, bidirectional=True, dropout=dropout)
                self.hid4_to_lat = nn.Linear(4 * nhid, nhid)

                self.enc = self.lstm_funct
            elif enc_type == 'bow':
                self.enc = nn.Linear(ninp, lat_dim)
            else:
                raise NotImplementedError
        elif self.dist_type == 'zero':
            self.enc = self.baseline_function
        else:
            raise NotImplementedError

        # VAE latent part
        if args.dist == 'nor':
            self.dist = Gauss(nhid, lat_dim)  # 2 for bidirect, 2 for h and
        elif args.dist == 'vmf':
            self.dist = vMF(nhid, lat_dim, kappa=self.args.kappa)
        elif args.dist == 'sph':
            pass
        elif args.dist == 'zero':
            pass
        else:
            raise NotImplementedError

        # LSTM
        self.z_to_h = nn.Linear(lat_dim, nhid * nlayers)  # TODO
        self.z_to_c = nn.Linear(lat_dim, nhid * nlayers)
        if args.fly:
            self.decoder_rnn = nn.LSTMCell(ninp + nhid, nhid, nlayers)
        else:
            self.decoder_rnn = nn.LSTM(ninp + lat_dim, nhid, nlayers, dropout=dropout)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder_out.weight = self.emb.weight

    def lstm_funct(self, x):
        batch_sz = x.size()[1]
        output, (h_n, c_n) = self.enc_lstm(x)
        concated_h_c = torch.cat((h_n[0], h_n[1], c_n[0], c_n[1]), dim=1)  # TODO
        # H = concated_h_c.permute(1, 0, 2).contiguous().view(batch_sz, 4 * self.nhid)
        return self.hid4_to_lat(concated_h_c)

    def baseline_function(self, x):
        seq_len, batch_sz = x.size()
        return torch.transpose(x, 1, 0)

    def dropword(self, emb, drop_rate=0.3):
        if self.FLAG_train:
            UNKs = GVar(torch.ones(emb.size()[0], emb.size()[1]).long() * 2)
            UNKs = self.emb(UNKs)
            # print(UNKs, emb)
            masks = numpy.random.binomial(1, drop_rate, size=(emb.size()[0], emb.size()[1]))
            masks = GVar(torch.FloatTensor(masks)).unsqueeze(2).expand_as(UNKs)
            emb = emb * (1 - masks) + UNKs * masks
            return emb
        else:
            return emb

    def forward(self, inp, target):
        seq_len, batch_sz = inp.size()
        tup, kld, decoded = self.run(inp, target)
        return tup, kld, decoded

    def forward_ground(self, inp, target):
        emb = self.drop(self.emb(inp))
        h = self.forward_enc(emb)
        tup, kld, z = self.forward_build_lat(h)  # batchsz, lat dim
        decoded = self.forward_decode_ground(emb, z)  # (seq_len, batch, dict sz)

        return tup, kld, decoded

    def forward_fly(self):
        pass

    def forward_enc(self, inp):
        """
        Given sequence, encode and yield a representation with hid_dim
        :param inp:
        :return:
        """
        # emb = self.drop(self.emb(inp))  # seq, batch, inp_dim
        h = self.enc(inp)
        return h

    def forward_build_lat(self, hidden):
        # hidden: batch_sz, nhid
        if self.args.dist == 'nor':
            tup, kld, out = self.dist.build_bow_rep(hidden, 1)  # 2 for bidirect, 2 for h and

        elif self.args.dist == 'vmf':
            tup, kld, out = self.dist.build_bow_rep(hidden, 1)

        elif self.args.dist == 'sph':
            norms = torch.norm(hidden, p=2, dim=1, keepdim=True)
            out = hidden / norms
            tup = {}
            kld = GVar(torch.zeros(1))
        elif self.args.dist == 'zero':
            out = GVar(torch.zeros(hidden.size()[0], self.lat_dim))
            tup = {}
            kld = GVar(torch.zeros(1))
        else:
            raise NotImplementedError
        return tup, kld, out

    def forward_decode_ground(self, emb, lat_code):
        # emb: seq, batch, ninp
        # latcode : batch, nlat
        seq_len, batch_sz, _ = emb.size()

        # Dropword
        emb = self.dropword(emb)
        lat_to_cat = lat_code.unsqueeze(0).expand(seq_len, batch_sz, -1)
        emb = torch.cat([emb, lat_to_cat], dim=2)

        # convert z to init h and c
        # (num_layers * num_directions, batch, hidden_size)
        init_h, init_c = self.convert_z_to_hidden(lat_code, batch_sz)
        # print(init_c.size(), init_h.size())
        # print(emb.size())
        # print(self.decoder_rnn)
        output, hidden = self.decoder_rnn(emb, (init_h, init_c))

        # output.size       (seq_len, batch, hidden_size)
        output = self.drop(output)
        decoded = self.decoder_out(output.view(output.size(0) * output.size(1), output.size(2)))
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded

    def forward_decode_fly(self, lat_code):
        pass

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = GVar(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def blstm_enc(self, input):
        """
        Encoding the input
        :param input: input sequence
        :return:
        embedding: seq_len, batch_sz, hid_dim
        hidden(from z): (2, batch_sz, 150)
        mu          : batch_sz, hid_dim
        logvar      : batch_sz, hid_dim
        """
        batch_sz = input.size()[1]
        emb = self.drop(self.emb(input))
        if self.dist == 'nor':
            mu, logvar = self.encode(emb)
            z = self.reparameterize(mu, logvar)  # z: batch, hid_dim

            hidden = self.convert_z_to_hidden(z, batch_sz)
            return emb, hidden, mu, logvar
        elif self.dist == 'vmf':
            mu = self.encode(emb)
            mu = mu.cpu()
            z = self.vmf.sample_vMF(mu)
            z = z.cuda()

            hidden = self.convert_z_to_hidden(z, batch_sz)
            return emb, hidden, mu
        else:
            raise NotImplementedError

    def encode(self, emb):
        """

        :param emb:
        :return: batch_sz, lat_dim
        """
        batch_sz = emb.size()[1]
        # self.enc_rnn.flatten_parameters()
        _, hidden = self.enc_rnn(emb)
        # num_layers * num_directions, batch, hidden_size
        h = hidden[0]
        c = hidden[1]
        assert h.size()[0] == self.nlayers * 2
        assert h.size()[1] == batch_sz
        x = torch.cat((h, c), dim=0).permute(1, 0, 2).contiguous().view(batch_sz, -1)
        if self.dist == 'nor':
            return self.fc_mu(x), self.fc_logvar(x)
        elif self.dist == 'vmf':
            return self.fc(x)
        else:
            raise NotImplementedError


    def forward_decode(self, args, input, ntokens):
        """

        :param args:
        :param input: LongTensor [seq_len, batch_sz]
        :param ntokens:
        :return:
            outputs_prob:   Var         seq_len, batch_sz, ntokens
            outputs:        LongTensor  seq_len, batch_sz
            mu, logvar
        """
        seq_len = input.size()[0]
        batch_sz = input.size()[1]

        emb, lat, mu, logvar = self.blstm_enc(input)
        # emb: seq_len, batchsz, hid_dim
        # hidden: ([2(nlayers),10(batchsz),200],[])

        outputs_prob = GVar(torch.FloatTensor(seq_len, batch_sz, ntokens))
        if args.cuda:
            outputs_prob = outputs_prob.cuda()

        outputs = torch.LongTensor(seq_len, batch_sz)

        # First time step sos
        sos = GVar(torch.ones(batch_sz).long())
        unk = GVar(torch.ones(batch_sz).long()) * 2
        if args.cuda:
            sos = sos.cuda()
            unk = unk.cuda()

        lat_to_cat = lat[0][0].unsqueeze(0)
        emb_t = self.drop(self.encoder(unk)).unsqueeze(0)
        emb_0 = self.drop(self.encoder(sos)).unsqueeze(0)
        emb_t_comb = torch.cat([emb_t, lat_to_cat], dim=2)
        emt_0_comb = torch.cat([emb_0, lat_to_cat], dim=2)

        hidden = None

        for t in range(seq_len):
            # input (seq_len, batch, input_size)

            if t == 0:
                emb = emt_0_comb
            else:
                emb = emb_t_comb
            # print(emb.size())

            if hidden is None:
                output, hidden = self.rnn(emb, None)
            else:
                output, hidden = self.rnn(emb, hidden)

            output_prob = self.decoder(self.drop(output))
            output_prob = output_prob.squeeze(0)
            outputs_prob[t] = output_prob
            value, ind = torch.topk(output_prob, 1, dim=1)
            outputs[t] = ind.squeeze(1).data

        return outputs_prob, outputs, mu, logvar

    def convert_z_to_hidden(self, z, batch_sz):
        """

        :param z:   batch, lat_dim
        :param batch_sz:
        :return:
        """
        h = self.z_to_h(z).view(batch_sz, self.nlayers, -1).permute(1, 0, 2).contiguous()
        c = self.z_to_c(z).view(batch_sz, self.nlayers, -1).permute(1, 0, 2).contiguous()
        return (h, c)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (GVar(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    GVar(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return GVar(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def init_weights(self):
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.enc_rnn.bias.data.fill_(0)
        self.enc_rnn.weight.data.uniform_(-initrange, initrange)
