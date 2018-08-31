import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Variable as Var
from archive.vae_proto import vMF


class VAEModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, args, dec_type, ntoken, ninp, nhid, lat_dim, nlayers, dropout=0.5, tie_weights=False):
        super(VAEModel, self).__init__()
        self.args = args
        self.lat_dim = lat_dim
        self.nhid = nhid
        self.nlayers = nlayers
        self.ninp = ninp
        self.dist = args.dist
        # VAE shared param
        self.emb = nn.Embedding(ntoken, ninp)

        # VAE recognition part
        # BLSTM
        self.enc_rnn = nn.LSTM(ninp, nhid, nlayers, bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)  # Need explicit dropout
        if args.dist == 'nor':
            self.fc_mu = nn.Linear(2 * nhid * nlayers * 2, lat_dim)  # 2 for bidirect, 2 for h and c
            self.fc_logvar = nn.Linear(2 * nhid * nlayers * 2, lat_dim)
        elif args.dist == 'vmf':
            self.fc = nn.Linear(2 * nhid * nlayers * 2, lat_dim)
            self.vmf = vMF.vmf(1, 10, args.kappa)
        else:
            raise NotImplementedError

        self.z_to_h = nn.Linear(lat_dim, nhid * nlayers)
        self.z_to_c = nn.Linear(lat_dim, nhid * nlayers)

        # VAE generation part
        self.dec_type = dec_type
        self.decoder_out = nn.Linear(nhid, ntoken)
        # LSTM
        if dec_type == 'lstm':

            if args.fly:
                self.decoder_rnn = nn.LSTMCell(ninp + nhid, nhid, nlayers)
            else:
                self.decoder_rnn = nn.LSTM(ninp + nhid, nhid, nlayers, dropout=dropout)

        # or
        # BoW
        elif dec_type == 'bow':
            self.linear = nn.Linear(nhid + ninp, nhid)
        else:
            raise NotImplementedError

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder_out.weight = self.emb.weight

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
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

    def forward(self, input):
        """

        :param input: seq_len, batch_sz
        :return:
        """

        batch_sz = input.size()[1]
        seq_len = input.size()[0]
        if self.dist == 'nor':
            emb, hidden, mu, logvar = self.blstm_enc(input)
        elif self.dist == 'vmf':
            emb, hidden, mu = self.blstm_enc(input)
            logvar = None
        if self.dec_type == 'lstm':
            lat_to_cat = hidden[0][0].unsqueeze(0).expand(seq_len, batch_sz, -1)
            emb = torch.cat([emb, lat_to_cat], dim=2)
            output, hidden = self.decoder_rnn(emb, hidden)
        elif self.dec_type == 'bow':
            # avg embedding: seq_len, batch_sz, hid_dim
            emb = torch.mean(emb, dim=0)  # torch.Size([20, 39])
            lat_to_cat = hidden[0][0]  # torch.Size([20, 49])
            fusion = torch.cat((emb, lat_to_cat), dim=1)  # torch.Size([20, 39+49])
            # output seq_len, batch, hidden_size * num_directions
            output = Variable(torch.FloatTensor(seq_len, batch_sz, self.nhid))
            if self.args.cuda:
                output = output.cuda()
            for t in range(seq_len):
                noise = 0.1 * Variable(fusion.data.new(fusion.size()).normal_(0, 1))
                if self.args.cuda:
                    noise = noise.cuda()
                fusion_with_noise = fusion + noise
                fusion_with_noise = self.linear(fusion_with_noise)
                output[t] = fusion_with_noise
        else:
            raise NotImplementedError

        output = self.drop(output)
        decoded = self.decoder_out(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), mu, logvar

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

        outputs_prob = Var(torch.FloatTensor(seq_len, batch_sz, ntokens))
        if args.cuda:
            outputs_prob = outputs_prob.cuda()

        outputs = torch.LongTensor(seq_len, batch_sz)

        # First time step sos
        sos = Var(torch.ones(batch_sz).long())
        unk = Var(torch.ones(batch_sz).long()) * 2
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
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def init_weights(self):
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.enc_rnn.bias.data.fill_(0)
        self.enc_rnn.weight.data.uniform_(-initrange, initrange)
