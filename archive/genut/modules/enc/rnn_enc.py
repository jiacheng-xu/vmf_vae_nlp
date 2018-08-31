import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var


class RNNEncoder(nn.Module):
    def __init__(self, opt, input_size, hidden_size, rnn_type='lstm'):
        super(RNNEncoder, self).__init__()

        # RNN Init
        self.n_layers = opt.enc_layers
        self.bidirect = True  # By default Bidirect
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn_type = rnn_type

        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True,
                              dropout=opt.dropout, bidirectional=self.bidirect)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                               dropout=opt.dropout, bidirectional=self.bidirect)

        else:
            raise NotImplementedError
        #
        self.init_weight()
        self.use_cuda = opt.use_cuda

    def init_weight(self):
        # kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros'
        if self.rnn_type == 'lstm':

            nn.init.xavier_uniform(self.rnn.weight_hh_l0, gain=1)
            nn.init.xavier_uniform(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform(self.rnn.weight_ih_l0, gain=1)
            nn.init.xavier_uniform(self.rnn.weight_ih_l0_reverse, gain=1)

            torch.nn.init.constant(self.rnn.bias_hh_l0, 0)
            nn.init.constant(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant(self.rnn.bias_ih_l0, 0)
            nn.init.constant(self.rnn.bias_ih_l0_reverse, 0)
        elif self.rnn_type == 'gru':
            nn.init.xavier_uniform(self.rnn.weight.data, gain=1)

    def init_hidden(self, batch_size):
        if self.bidirect:
            result = Var(torch.zeros(self.n_layers * 2, batch_size, self.hidden_size))
        else:
            result = Var(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result

    def forward(self, embedded, inp_msk):
        """

        :param input: (seq_len, batch size, inp_dim)
        :param inp_msk: [seq len,....]

        :return: output: PackedSequence (seq_len*batch,  hidden_size * num_directions),
                hidden tupple ((batch, hidden_size*2), ....)
        """
        batch_size = embedded.data.shape[0]

        # list_msk = torch.sum(inp_msk.long(), dim=0).tolist()

        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded, inp_msk, batch_first=True)

        if self.rnn_type == 'lstm':
            # output, hn = self.rnn(packed_embedding,
            #                       (self.init_hidden(batch_size), self.init_hidden(batch_size)))
            output, hn = self.rnn(packed_embedding)
        else:
            output, hn = self.rnn(packed_embedding, self.init_hidden(batch_size))

        def _fix_hidden(hidden):
            """

            :param hidden: (num_directions, batch, hidden_size)
            :return: batch, hidden_size*2
            """
            hidden = torch.cat((hidden[0], hidden[1]), dim=1)
            # hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
            return hidden

        def compress_blstm_hidden(whole_context):
            # whole_context (seq_len, batch, hidden_size * num_directions)
            return F.relu(self.reduce_h_W(whole_context))

        output, context_mask_ = nn.utils.rnn.pad_packed_sequence(output)

        if self.bidirect:
            if self.rnn_type == 'lstm':
                output = compress_blstm_hidden(output)

                h, c = hn[0], hn[1]
                h_, c_ = _fix_hidden(h), _fix_hidden(c)
                # new_h = F.relu(self.reduce_h_W(h_))
                # new_c = F.relu(self.reduce_c_W(c_))
                new_h = self.reduce_h_W(h_)
                new_c = self.reduce_c_W(c_)
                h_t = (new_h, new_c)
            elif self.rnn_type == 'gru':
                h_t = (_fix_hidden(hn))
        else:
            raise NotImplementedError
        return h_t


class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == "__main__":
    unittest.main()
    batch = 16
    seq = 200
    dim = 400
