import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.autograd import Variable as Var

from archive.genut import Attention


class RNNDecoder(nn.Module):
    def __init__(self, opt, rnn_type='lstm', num_layers=1,
                 hidden_size=100, input_size=50, attn_type='dot', coverage=False,
                 copy=False, dropout=0.1, emb=None, full_dict_size=None, word_dict_size=None,
                 max_len_dec=100, beam=True):
        super(RNNDecoder, self).__init__()
        self.opt = opt
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.word_dict_size = word_dict_size
        self.full_dict_size = full_dict_size
        self.embeddings = emb

        self.max_len_dec = max_len_dec
        self.rnn = self.build_rnn(rnn_type, self.input_size, hidden_size,
                                  num_layers)
        self.mask = None
        self.attn = Attention(opt, hidden_size, attn_type, coverage, opt.feat_nn_dim, opt.feat_sp_dim)
        self.W_out_0 = nn.Linear(hidden_size * 3, word_dict_size, bias=True)

        self.sampling = opt.schedule
        self._coverage = coverage

        self._copy = copy
        if copy:
            # self.sigmoid = nn.Sigmoid()
            self.copy_linear = nn.Linear(hidden_size * 3 + input_size, 1, bias=True)
            # self.copy_w_h = nn.Linear(hidden_size, 1, bias=True)  # batch, hid_dim
            # self.copy_w_s = nn.Linear(hidden_size, 1, bias=True)  # batch, hid_dim
            # self.copy_w_c = nn.Linear(hidden_size, 1, bias=True)  # batch, hid_dim
            # self.copy_w_x = nn.Linear(input_size, 1, bias=True)  # batch, inp_dim
            # self.copy_lb = copy_lb  # Copy lower bound
        # if copy_attn:
        #     self.copy_attn = Attention(
        #         hidden_size, attn_type=attn_type
        #     )
        #     self._copy = True

        if beam is True:
            self.beam_size = opt.beam_size
            assert self.beam_size >= 1

    # def set_mask(self, mask):
    #     raise NotImplementedError
    #     batch_size = len(mask)
    #     max_len = mask[0]
    #     mask_mat = np.ones((batch_size, max_len), dtype=int)
    #     for idx, i in enumerate(mask):
    #         mask_mat[idx, :i] = 0
    #     # Invalid position == 1!!!
    #     mask = torch.ByteTensor(mask_mat).cuda()
    #     self.mask = mask
    #     self.attn.mask = mask

    def build_rnn(self, rnn_type, input_size,
                  hidden_size, num_layers):
        if num_layers > 1:
            raise NotImplementedError
        if rnn_type == "lstm":
            return torch.nn.LSTMCell(input_size, hidden_size, bias=True)
        elif rnn_type == 'gru':
            return torch.nn.GRUCell(input_size, hidden_size, bias=True)
        else:
            raise NotImplementedError

    def run_forward_step(self, input, context, context_mask, feats, prev_state, prev_attn, coverage=None, inp_var=None,
                         max_oov_len=None,
                         scatter_mask=None):
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

        if isinstance(prev_state, (tuple)):
            batch_size_, hidden_size_ = prev_state[0].size()
        else:
            batch_size_, hidden_size_ = prev_state.size()
        src_len, batch_size, hidden_size = context.size()
        # input = input.squeeze()
        inp_size, batch_size__ = input.size()
        assert inp_size == 1
        assert batch_size_ == batch_size__ == batch_size

        # Start running
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

        attn_h_weighted, a = self.attn.forward(current_raw_state,
                                               context.clone().transpose(0, 1), context_mask, prev_attn,
                                               coverage, feats)

        # attn_h_weighted: batch, dim
        # a:               batch, src

        if self._copy:
            # copy_merge =
            copy_mat = self.copy_linear(torch.cat([attn_h_weighted, current_state_h, current_state_c, emb], dim=1))
            p_gen = F.sigmoid(copy_mat)

        hidden_state_vocab = torch.cat([attn_h_weighted, current_state_h, current_state_c], 1)
        hidden_state_vocab = self.W_out_0(hidden_state_vocab)
        # prob_vocab = F.softmax(hidden_state_vocab)

        max_hidden_state_vocab = torch.max(hidden_state_vocab,
                                           dim=1, keepdim=True)[0]
        hidden_state_vocab = hidden_state_vocab - max_hidden_state_vocab
        prob_vocab = F.softmax(hidden_state_vocab)  # prob over vocab
        # print('prob_vocab')
        # print(prob_vocab)
        if self._copy:
            prob_vocab = p_gen * prob_vocab
            new_a = a.clone()
            # Scatter
            zeros = Var(torch.zeros(batch_size_, self.word_dict_size + max_oov_len))
            assert a.size()[1] == src_len
            assert inp_var.size()[0] == src_len
            assert inp_var.size()[1] == batch_size

            # print(new_a)
            # print(scatter_mask)
            # exit()
            # notice: t1.size(1) is equal orig.size(1)
            # prob_copy = Var(zeros.scatter_(1, inp_var.data.transpose(1, 0).cpu(), a.data.cpu())).cuda()
            # Copy scatter
            # y = torch.sum(new_a)
            new_attn = torch.bmm(scatter_mask, new_a.unsqueeze(2)).squeeze(2)

            prob_copy = zeros.scatter_(1, inp_var.transpose(1, 0).cpu(), new_attn.cpu()).cuda()
            # x = torch.sum(prob_copy,dim=1)
            # print(x)
            prob_final = Var(torch.zeros(prob_copy.size())).cuda()

            prob_final[:, :self.opt.word_dict_size] = prob_vocab
            prob_final = prob_final + (1 - p_gen) * prob_copy
        else:
            p_gen = 1
            zeros = torch.zeros(batch_size_, self.word_dict_size + max_oov_len)
            prob_final = Var(torch.zeros(zeros.size())).cuda()
            prob_final[:, :self.opt.word_dict_size] = prob_vocab

        prob_final = torch.log(prob_final + 0.0000001)

        return current_raw_state, prob_final, coverage, a, p_gen
        # current_state_h is raw hidden before attention
        # prob_final is the final probability

    def forward(self, context, inp_msk,
                h_t,
                tgt_var, tgt_msk,
                inp_var, aux):
        """

        :param context:
        :param inp_msk:
        :param h_t:
        :param tgt_var:
        :param tgt_msk:
        :param inp_var:
        :param aux:
        :return:
        """
        # context, context_mask, state, tgt, tgt_mask, inp_var, feat, max_oov_len, scatter_mask,
        # bigram_bunch,
        # logger):
        # if self.opt.mul_loss or self.opt.add_loss:        # TODO additional module unavail
        #     bigram, bigram_msk, bigram_dicts = bigram_bunch
        #     bigram = Var(bigram, requires_grad=False).cuda()
        #     # bigram_msk = Var(bigram_msk, requires_grad=False).cuda()
        #
        #     # window_msk = Var(window_msk, requires_grad=False).float().cuda().contiguous()

        tgt_len, batch_size = tgt.size()

        inp_var = inp_var[0]

        # assert context_mask == context_mask
        # contxt_len, batch_size_, _ = context.size()
        src_len, batch_size_, hidden_size = context.size()
        assert batch_size_ == batch_size

        # self.set_mask(context_mask)
        # padding_mask = util.mask_translator(context_mask, batch_first=True, is_var=True)
        padding_mask = context_mask.transpose(1, 0)

        decoder_outputs = torch.LongTensor(tgt_len, batch_size)
        decoder_outputs_prob = Var(torch.zeros((tgt_len, batch_size, self.word_dict_size + max_oov_len))).cuda()

        decoder_input = np.ones((1, batch_size), dtype=int) * self.opt.sos
        decoder_input = Var(torch.LongTensor(decoder_input))

        loss_cov = Var(torch.zeros(tgt_len, batch_size)).cuda()

        if self._copy:
            p_copys = torch.zeros((batch_size, tgt_len)).cuda()

        if self._coverage:
            coverage = Var(torch.zeros((batch_size, src_len))).cuda() + 0.0001
        else:
            coverage = None

        attn = Var(torch.zeros((batch_size, src_len))).cuda()
        Attns = Var(torch.zeros((tgt_len, batch_size, src_len))).cuda()

        # Discount = Var(torch.zeros((tgt_len, batch_size))).cuda()  # Discount over attention
        # Discount = Var(
        #     torch.zeros((tgt_len, batch_size, self.word_dict_size + max_oov_len))).cuda()  # Discount over output

        # if self.opt.mul_loss or self.opt.add_loss:
        #     Table = Var(torch.zeros((batch_size_, src_len, src_len))).cuda()
        # else:
        #     Table = None

        for t in range(tgt_len):
            state, prob_final, coverage, attn, p_gen = \
                self._run_forward_one(decoder_input, context, padding_mask, feat, state, attn, coverage, inp_var,
                                      max_oov_len,
                                      scatter_mask)

            if self._copy:
                p_copys[:, t] = p_gen.data

            decoder_outputs_prob[t] = prob_final

            topv, topi = prob_final.data.topk(1)
            topi = topi.squeeze()

            # Record
            decoder_outputs[t] = topi
            Attns[t] = attn

            if self.opt.mul_loss or self.opt.add_loss:
                """
                        This function supports bi-gram pattern matching between txt and abs.
                        eg Input: A B A C A D
                            Out: A B A D
                        :param txt: src_seq, batch
                        :param abs: tgt_seq, batch
                        bigram_msk, repeat_map, window_msk,
                        :return: 1) batch_sz, seq_len, seq_len  records all the bigrams information. the first seq_len means the prev word.
                                                            the second one denotes the location where there are bigrams.
                                                            A[ 0 1 0 0 0 1]  note that C position is 0 since AC doesn't appear in gold summary
                        bigram_msk                         B[ 1 0 0 0 0 0]
                                                            A[ 0 1 0 0 0 1]
                                                            D[ 0 0 0 0 0 0 ]
                                2)  batch_sz, tgt_len           Records all 
                                                        [ 0 , 0 , 1 , 0 ] the first 0 is default zero. 
                             repeat_map                  the second 0 means 'look at B, prev is A, A is a word in txt, 0 is the first time the location it appears'
                                                        if the prefix word doesn't appear in the doc, it is 0. (3) will filter out this
                                3)  batch_sz, tgt_len       accordingly cooperates with (2). val =1 if prefix word appear in the document
                                window_msk
                """

                """
                New Version!
                tgt_msk = current_batch['bigram_msk']
                dict_from_wid_to_pos_in_tgt = current_batch['bigram_dict']

                current_batch['bigram']     batchsz, tgtlen, srclen
                current_batch['bigram_msk'] batchsz, tgtlen
                current_batch['bigram_dict']batchsz,   Not usable now # TODO

                """
                decoder_input = decoder_input.squeeze(0)
                for bidx in range(batch_size_):
                    prev = decoder_input[bidx].data[0]
                    # print(prev)
                    _dict = bigram_dicts[bidx]
                    if _dict.has_key(prev):
                        v = _dict[prev]
                        if v < tgt_len:
                            _bigram = bigram[bidx][v]
                            # x = torch.sum(_bigram[v] * attn[bidx])   # Discount over attention
                            _sum = torch.sum(_bigram)
                            if (_sum.data > 0).all():
                                tmp = _bigram
                                # print(tmp.size())
                                # print(inp_var[:, bidx].size())
                                # print(max_oov_len)
                                # print(torch.max(inp_var[:, bidx]))
                                # exit()
                                zeros = Var(torch.zeros(self.word_dict_size + max_oov_len))
                                x = zeros.scatter_(0, inp_var[:, bidx].cpu(),
                                                   tmp.cpu()).cuda()
                            else:
                                x = 0
                    else:
                        x = 0

                    Discount[t, bidx] = x
                """
                window = window_msk[:, t].contiguous()
                repeat = repeat_map[:, t]
                tables = Table[range(batch_size_), repeat, :]
                x = torch.min(attn, 1 - tables)  # accumulative attention matrix for
                y = bigram_msk[range(batch_size_), repeat, :]
                z = x * y
                z = z * window.view(-1, 1)
                zeros = torch.zeros(batch_size, self.word_dict_size + max_oov_len)
                accumulated_z = torch.bmm(scatter_mask, z.unsqueeze(2)).squeeze(2)
                z = zeros.scatter_(1, inp_var.transpose(1, 0).data.cpu(), accumulated_z.data.cpu()).cuda()
                Discount[t] = z
                update = torch.min(Table[range(batch_size_), repeat, :] + attn,
                                   Var(torch.ones((batch_size_, src_len))).cuda())
                Table[range(batch_size_), repeat] = update
                """

            if random.random() >= self.sampling:
                decoder_input = topi
                decoder_input = Var(decoder_input.unsqueeze(0))
            else:
                decoder_input = Var(tgt[t]).unsqueeze(0)

            # Compute Coverage Loss
            if self._coverage:
                merged_attn_coverage = torch.cat((attn.unsqueeze(2), coverage.unsqueeze(2)), dim=2)
                merge_min = torch.min(merged_attn_coverage, 2)  # bug
                loss_cov[t, :] = torch.sum(merge_min[0], dim=1)

                coverage = coverage + attn

        return decoder_outputs_prob, decoder_outputs, Attns, Discount, loss_cov, p_copys
