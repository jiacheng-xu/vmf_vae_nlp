import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.autograd import Variable as Var


class SimpleRNNDecoder(nn.Module):
    """
    Simple Recurrent Module without attn, copy, coverage.
    """

    def __init__(self, opt, rnn_type, input_size, hidden_size, emb):
        super(SimpleRNNDecoder, self).__init__()
        self.opt = opt
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.full_dict_size = opt.full_dict_size
        self.embeddings = emb

        self.rnn = self.build_rnn(rnn_type, self.input_size, hidden_size)
        self.W_out_0 = nn.Linear(hidden_size * 2, opt.full_dict_size, bias=True)

    def build_rnn(self, rnn_type, input_size,
                  hidden_size, num_layers=1):
        if num_layers > 1:
            raise NotImplementedError
        if rnn_type == "lstm":
            return torch.nn.LSTMCell(input_size, hidden_size, bias=True)
        elif rnn_type == 'gru':
            return torch.nn.GRUCell(input_size, hidden_size, bias=True)
        else:
            raise NotImplementedError

    def run_forward_step(self, input, prev_state):
        """
        :param input: (LongTensor): a sequence of input tokens tensors
                                of size (1 x batch).
        :param context: (FloatTensor): output(tensor sequence) from the enc
                        RNN of size (src_len x batch x hidden_size).
        :param prev_state: tuple (FloatTensor): Maybe a tuple if lstm. (batch x hidden_size) hidden state from the enc RNN for
                                 initializing the decoder.
        :param coverage
        :param inp_var
        :return:
        """
        # print(input)
        # print(prev_state)
        if isinstance(prev_state, (tuple)):
            batch_size_, hidden_size_ = prev_state[0].size()
        else:
            batch_size_, hidden_size_ = prev_state.size()

        inp_size, batch_size__ = input.size()

        assert inp_size == 1
        assert batch_size__ == batch_size_

        # Start running
        if self.opt.use_cuda:
            input = input.cuda()

        emb = self.embeddings.forward_decoding(input)
        assert emb.dim() == 3  # 1 x batch x embedding_dim
        emb = emb.squeeze(0)  # batch, embedding_dim
        #
        # print(emb.size())
        # print(prev_state.size())
        # print(self.rnn)
        current_raw_state = self.rnn(emb, prev_state)  # rnn_output: batch x hiddensize. hidden batch x hiddensize

        if self.rnn_type == 'lstm':
            assert type(current_raw_state) == tuple
            current_state_h = current_raw_state[0]
            current_state_c = current_raw_state[1]
        elif self.rnn_type == 'gru':
            current_state_h = current_raw_state
            current_state_c = current_raw_state
        else:
            raise NotImplementedError

        hidden_state_vocab = torch.cat([current_state_h, current_state_c], 1)
        hidden_state_vocab = self.W_out_0(hidden_state_vocab)
        # prob_vocab = F.softmax(hidden_state_vocab)

        # max_hidden_state_vocab = torch.max(hidden_state_vocab,
        #                                    dim=1, keepdim=True)[0]
        # hidden_state_vocab = hidden_state_vocab - max_hidden_state_vocab
        # prob_vocab = F.softmax(hidden_state_vocab)  # prob over vocab
        #
        # prob_final = torch.log(prob_vocab)

        return current_raw_state, hidden_state_vocab
        # current_state_h is raw hidden before attention
        # prob_final is the final probability

    def forward(self,
                state,
                tgt_var, tgt_msk,
                aux):
        batch_size, tgt_len = tgt_var.size()
        mode = self.training

        decoder_outputs = torch.LongTensor(tgt_len, batch_size)

        decoder_outputs_prob = Var(torch.zeros((tgt_len, batch_size, self.full_dict_size)))
        if self.opt.use_cuda:
            decoder_outputs_prob = decoder_outputs_prob.cuda()

        decoder_input = np.ones((1, batch_size), dtype=int) * self.opt.unk
        decoder_input = Var(torch.LongTensor(decoder_input))

        for t in range(tgt_len):
            state, prob_final = \
                self.run_forward_step(decoder_input, state)

            decoder_outputs_prob[t] = prob_final

            topv, topi = prob_final.data.topk(1)
            topi = topi.squeeze()
            # Record
            decoder_outputs[t] = topi

            if mode:  # train mode
                if random.random() >= self.opt.schedule:
                    decoder_input = topi
                    decoder_input = Var(decoder_input.unsqueeze(0))
                    # print(decoder_input.size())
                else:
                    decoder_input = Var(tgt_var[:, t]).unsqueeze(0)
            else:  # eval mode
                decoder_input = Var(tgt_var[:, t]).unsqueeze(0)

        return decoder_outputs_prob, decoder_outputs

    def init_weight(self):
        # kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros'
        if self.rnn_type == 'lstm':

            nn.init.xavier_uniform(self.rnn.weight_ih, gain=1)
            nn.init.xavier_uniform(self.rnn.weight_hh, gain=1)

            torch.nn.init.constant(self.rnn.bias_ih, 0)
            nn.init.constant(self.rnn.bias_hh, 0)
        elif self.rnn_type == 'gru':
            nn.init.xavier_uniform(self.rnn.weight.data, gain=1)
