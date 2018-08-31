import torch
import torch.nn as nn
from torch.autograd import Variable


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, agenda_dim, nlayers=1, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(ntoken, ninp)

        self.decoder_rnn = nn.LSTM(ninp + agenda_dim, nhid, nlayers, dropout=dropout)
        self.decoder_out = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder_out.weight = self.embed.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):

        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.decoder_out.bias.data.fill_(0)
        self.decoder_out.weight.data.uniform_(-initrange, initrange)

        # # kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'
        # torch.nn.init.xavier_uniform(self.decoder_rnn.weight_ih_l0.data, gain=nn.init.calculate_gain('sigmoid'))
        # torch.nn.init.orthogonal(self.decoder_rnn.weight_hh_l0.data, gain=nn.init.calculate_gain('sigmoid'))
        #
        # # embedding uniform
        # torch.nn.init.xavier_uniform(self.embed.weight.data, gain=nn.init.calculate_gain('linear'))
        #
        # # Linear kernel_initializer='glorot_uniform'
        # torch.nn.init.xavier_uniform(self.decoder_out.weight.data, gain=nn.init.calculate_gain('linear'))

    def forward(self, input, hidden=None):
        batch_sz = input.size()[1]
        if hidden is None:
            hidden = self.init_hidden(batch_sz)
        emb = self.drop(self.embed(input))
        output, hidden = self.decoder_rnn(emb, hidden)
        output = self.drop(output)
        # output (seq_len, batch, hidden_size * num_directions)
        decoded = self.decoder_out(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def forward_decode(self, args, input, ntokens):

        seq_len = input.size()[0]
        batch_sz = input.size()[1]
        # emb: seq_len, batchsz, hid_dim
        # hidden: ([2(nlayers),10(batchsz),200],[])
        hidden = None
        outputs_prob = Variable(torch.FloatTensor(seq_len, batch_sz, ntokens))
        if args.cuda:
            outputs_prob = outputs_prob.cuda()
        outputs = torch.LongTensor(seq_len, batch_sz)

        # First time step sos
        sos = Variable(torch.ones(batch_sz).long())  # id for sos =1
        unk = Variable(torch.ones(batch_sz).long()) * 2  # id for unk =2
        if args.cuda:
            sos = sos.cuda()
            unk = unk.cuda()

        emb_0 = self.drop(self.encoder(sos)).unsqueeze(0)
        emb_t = self.drop(self.encoder(unk)).unsqueeze(0)

        for t in range(seq_len):
            # input (seq_len, batch, input_size)
            if t == 0:
                emb = emb_0
            else:
                emb = emb_t

            output, hidden = self.rnn(emb, hidden)
            output_prob = self.decoder(self.drop(output))
            output_prob = output_prob.squeeze(0)
            outputs_prob[t] = output_prob
            value, ind = torch.topk(output_prob, 1, dim=1)
            outputs[t] = ind.squeeze(1).data
        return outputs_prob, outputs

    def init_hidden(self, bsz):
        return (Variable(torch.zeros(self.nlayers, bsz, self.nhid)).cuda(),
                Variable(torch.zeros(self.nlayers, bsz, self.nhid)).cuda())
        # weight = next(self.parameters()).data
        # if self.rnn_type == 'LSTM':
        #     return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
        #             Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        # else:
        #     return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
