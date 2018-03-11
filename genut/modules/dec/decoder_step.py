import torch
import torch.nn.functional as F
import torch.optim
from torch.autograd import Variable as Var

from Decoder import RNNDecoderBase


class InputFeedRNNDecoder(RNNDecoderBase):
    def _run_forward_one(self, input, context, context_mask, feats, prev_state, prev_attn, coverage=None, inp_var=None,
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
