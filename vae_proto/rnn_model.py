import torch
import torch.nn as nn
from torch.autograd import Variable


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
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

    def forward(self, input, hidden=None):
        batch_sz = input.size()[1]
        if hidden is None:
            hidden = self.init_hidden(batch_sz)
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
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
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
