import numpy
import torch
import torch.nn as nn

from NVLL.distribution.gauss import Gauss
from NVLL.distribution.vmf_batch import vMF
from NVLL.distribution.vmf_unif import unif_vMF
from NVLL.distribution.vmf_hypvae import VmfDiff
from NVLL.util.util import GVar
from NVLL.util.util import check_dispersion

numpy.random.seed(2018)


class RNNVAE(nn.Module):
    """Container module with an optional encoder, a prob latent module, and a RNN decoder."""

    def __init__(self, args, enc_type, ntoken, ninp, nhid,
                 lat_dim, nlayers, dropout=0.5, tie_weights=False,
                 input_z=False, mix_unk=0, condition=False, input_cd_bow=0, input_cd_bit=0):
        assert (not condition) or (condition and (input_cd_bow > 1 or input_cd_bit > 1))
        assert type(input_cd_bit) == int and input_cd_bit >= 0
        assert type(input_cd_bow) == int and input_cd_bow >= 0

        super(RNNVAE, self).__init__()
        self.FLAG_train = True
        self.args = args
        self.enc_type = enc_type
        print("Enc type: {}".format(enc_type))
        try:
            self.bi = args.bi
        except AttributeError:
            self.bi = True

        self.input_z = input_z
        self.condition = condition
        self.input_cd_bow = input_cd_bow
        self.input_cd_bit = input_cd_bit

        self.lat_dim = lat_dim
        self.nhid = nhid
        self.nlayers = nlayers  # layers for decoding part
        self.ninp = ninp
        self.ntoken = ntoken
        self.dist_type = args.dist  # Gauss or vMF = normal vae;
        # zero = no encoder and VAE, use 0 as start of decoding;
        #  sph = encode word embedding as bow and project to a unit sphere

        # VAE shared param
        self.drop = nn.Dropout(dropout)  # Need explicit dropout
        self.emb = nn.Embedding(ntoken, ninp)
        if input_cd_bit > 1:
            self.emb_bit = nn.Embedding(5, input_cd_bit)
        if input_cd_bow > 1:
            self.nn_bow = nn.Linear(ninp, input_cd_bow)

        # VAE decoding part
        self.decoder_out = nn.Linear(nhid, ntoken)

        # VAE recognition part
        if self.dist_type == 'nor' or 'vmf' or 'sph' or 'unifvmf':
            _factor = 1
            _inp_dim = ninp
            if input_cd_bit > 1:
                _inp_dim += int(input_cd_bit)
            if (enc_type == 'lstm') or (enc_type == 'gru'):
                if enc_type == 'lstm':
                    _factor *= 2
                    self.enc_rnn = nn.LSTM(_inp_dim, nhid, 1, bidirectional=self.bi, dropout=dropout)
                elif enc_type == 'gru':
                    self.enc_rnn = nn.GRU(_inp_dim, nhid, 1, bidirectional=self.bi, dropout=dropout)
                else:
                    raise NotImplementedError
                if self.bi:
                    _factor *= 2

                self.hid4_to_lat = nn.Linear(_factor * nhid, nhid)

                self.enc = self.rnn_funct
            elif enc_type == 'bow':
                self.enc = self.bow_funct
                self.hid4_to_lat = nn.Linear(ninp, nhid)
            else:
                raise NotImplementedError
        elif self.dist_type == 'zero':
            pass
        else:
            raise NotImplementedError

        # VAE latent part
        if args.dist == 'nor':
            self.dist = Gauss(nhid, lat_dim)  # 2 for bidirect, 2 for h and
        elif args.dist == 'vmf':
            self.dist = vMF(nhid, lat_dim, kappa=self.args.kappa)
        elif args.dist == 'sph':
            self.dist = VmfDiff(nhid, lat_dim)
        elif args.dist == 'zero':
            pass
        elif args.dist == 'unifvmf':
            self.dist = unif_vMF(nhid, lat_dim,
                                 kappa=self.args.kappa, norm_max=self.args.norm_max
                                 )
        else:
            raise NotImplementedError

        self.mix_unk = mix_unk

        # LSTM
        self.z_to_h = nn.Linear(lat_dim, nhid * nlayers)
        self.z_to_c = nn.Linear(lat_dim, nhid * nlayers)

        _dec_rnn_inp_dim = ninp
        if input_z:
            _dec_rnn_inp_dim += lat_dim
        if input_cd_bit > 1:
            _dec_rnn_inp_dim += int(input_cd_bit)
        if input_cd_bow > 1:
            _dec_rnn_inp_dim += int(input_cd_bow)

        self.decoder_rnn = nn.LSTM(_dec_rnn_inp_dim, nhid, nlayers, dropout=dropout)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder_out.weight = self.emb.weight

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    def bow_funct(self, x):
        y = torch.mean(x, dim=0)

        y = self.hid4_to_lat(y)
        y = torch.nn.functional.tanh(y)
        return y

    def rnn_funct(self, x):
        batch_sz = x.size()[1]
        if self.enc_type == 'lstm':
            output, (h_n, c_n) = self.enc_rnn(x)
            if self.bi:
                concated_h_c = torch.cat((h_n[0], h_n[1], c_n[0], c_n[1]), dim=1)
            else:
                concated_h_c = torch.cat((h_n[0], c_n[0]), dim=1)
        elif self.enc_type == 'gru':
            output, h_n = self.enc_rnn(x)
            if self.bi:
                concated_h_c = torch.cat((h_n[0], h_n[1]), dim=1)
            else:
                concated_h_c = h_n[0]
        else:
            raise NotImplementedError
        return self.hid4_to_lat(concated_h_c)

    def dropword(self, emb, drop_rate=0.3):
        """
        Mix the ground truth word with UNK.
        If drop rate = 1, no ground truth info is used. (Fly mode)
        :param emb:
        :param drop_rate: 0 - no drop; 1 - full drop, all UNK
        :return: mixed embedding
        """
        UNKs = GVar(torch.ones(emb.size()[0], emb.size()[1]).long() * 2)
        UNKs = self.emb(UNKs)
        # print(UNKs, emb)
        masks = numpy.random.binomial(1, drop_rate, size=(emb.size()[0], emb.size()[1]))
        masks = GVar(torch.FloatTensor(masks)).unsqueeze(2).expand_as(UNKs)
        emb = emb * (1 - masks) + UNKs * masks
        return emb

    def forward(self, inp, target, bit=None):
        """
        Forward with ground truth (maybe mixed with UNK) as input.
        :param inp:  seq_len, batch_sz
        :param target: seq_len, batch_sz
        :param bit: 1, batch_sz
        :return:
        """
        seq_len, batch_sz = inp.size()
        emb = self.drop(self.emb(inp))

        if self.input_cd_bow > 1:
            bow = self.enc_bow(emb)
        else:
            bow = None
        if self.input_cd_bit > 1:
            bit = self.enc_bit(bit)
        else:
            bit = None

        h = self.forward_enc(emb, bit)
        tup, kld, vecs = self.forward_build_lat(h, self.args.nsample)  # batchsz, lat dim

        if 'redundant_norm' in tup:
            aux_loss = tup['redundant_norm'].view(batch_sz)
        else:
            aux_loss = GVar(torch.zeros(batch_sz))
        if 'norm' not in tup:
            tup['norm'] = GVar(torch.zeros(batch_sz))
        # stat
        avg_cos = check_dispersion(vecs)
        tup['avg_cos'] = avg_cos

        avg_norm = torch.mean(tup['norm'])
        tup['avg_norm'] = avg_norm

        vec = torch.mean(vecs, dim=0)

        decoded = self.forward_decode_ground(emb, vec, bit, bow)  # (seq_len, batch, dict sz)

        flatten_decoded = decoded.view(-1, self.ntoken)
        flatten_target = target.view(-1)
        loss = self.criterion(flatten_decoded, flatten_target)
        return loss, kld, aux_loss, tup, vecs, decoded

    def enc_bit(self, bit):
        if self.input_cd_bit > 1:
            return self.emb_bit(bit)
        else:
            return None

    def enc_bow(self, emb):
        if self.input_cd_bow > 1:
            x = self.nn_bow(torch.mean(emb, dim=0))
            return x
        else:
            return None

    def forward_enc(self, inp, bit=None):
        """
        Given sequence, encode and yield a representation with hid_dim
        :param inp:
        :return:
        """
        seq_len, batch_sz = inp.size()[0:2]
        # emb = self.drop(self.emb(inp))  # seq, batch, inp_dim
        if self.dist_type == 'zero':
            return torch.zeros(batch_sz)
        if bit is not None:
            bit = bit.unsqueeze(0).expand(seq_len, batch_sz, -1)
            inp = torch.cat([inp, bit], dim=2)
        h = self.enc(inp)
        # print(h.size())
        return h

    def forward_build_lat(self, hidden, nsample=3):
        """

        :param hidden:
        :return: tup, kld [batch_sz], out [nsamples, batch_sz, lat_dim]
        """
        # hidden: batch_sz, nhid
        if self.args.dist == 'nor':
            tup, kld, out = self.dist.build_bow_rep(hidden, nsample)  # 2 for bidirect, 2 for h and
        elif self.args.dist == 'vmf':
            tup, kld, out = self.dist.build_bow_rep(hidden, nsample)
        elif self.args.dist == 'unifvmf':
            tup, kld, out = self.dist.build_bow_rep(hidden, nsample)
        elif self.args.dist == 'vmf_diff':
            tup, kld, out = self.dist.build_bow_rep(hidden, nsample)
        elif self.args.dist == 'sph':
            tup, kld, out = self.dist.build_bow_rep(hidden, nsample)
        elif self.args.dist == 'zero':
            out = GVar(torch.zeros(1, hidden.size()[0], self.lat_dim))
            tup = {}
            kld = GVar(torch.zeros(1))
        else:
            raise NotImplementedError
        return tup, kld, out

    def forward_decode_ground(self, emb, lat_code, bit=None, bow=None):
        """

        :param emb: seq, batch, ninp
        :param lat_code: batch, nlat
        :param bit:
        :param bow:
        :return:
        """

        seq_len, batch_sz, _ = emb.size()

        # Dropword
        if self.mix_unk > 0.001:
            emb = self.dropword(emb, self.mix_unk)

        if self.input_z:
            lat_to_cat = lat_code.unsqueeze(0).expand(seq_len, batch_sz, -1)
            emb = torch.cat([emb, lat_to_cat], dim=2)

        if self.input_cd_bow > 1:
            bow = bow.unsqueeze(0).expand(seq_len, batch_sz, -1)
            emb = torch.cat([emb, bow], dim=2)

        if self.input_cd_bit > 1:
            bit = bit.unsqueeze(0).expand(seq_len, batch_sz, -1)
            emb = torch.cat([emb, bit], dim=2)

        # convert z to init h and c
        # (num_layers * num_directions, batch, hidden_size)
        init_h, init_c = self.convert_z_to_hidden(lat_code, batch_sz)
        output, hidden = self.decoder_rnn(emb, (init_h, init_c))

        # output.size       (seq_len, batch, hidden_size)
        output = self.drop(output)
        decoded = self.decoder_out(output.view(output.size(0) * output.size(1), output.size(2)))
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded

    # def blstm_enc(self, input):
    #     """
    #     Encoding the input
    #     :param input: input sequence
    #     :return:
    #     embedding: seq_len, batch_sz, hid_dim
    #     hidden(from z): (2, batch_sz, 150)
    #     mu          : batch_sz, hid_dim
    #     logvar      : batch_sz, hid_dim
    #     """
    #     batch_sz = input.size()[1]
    #     emb = self.drop(self.emb(input))
    #     if self.dist == 'nor':
    #         mu, logvar = self.encode(emb)
    #         z = self.reparameterize(mu, logvar)  # z: batch, hid_dim
    #
    #         hidden = self.convert_z_to_hidden(z, batch_sz)
    #         return emb, hidden, mu, logvar
    #     elif self.dist == 'vmf':
    #         mu = self.encode(emb)
    #         mu = mu.cpu()
    #         z = self.vmf.sample_vMF(mu)
    #         z = z.cuda()
    #
    #         hidden = self.convert_z_to_hidden(z, batch_sz)
    #         return emb, hidden, mu
    #     else:
    #         raise NotImplementedError

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
