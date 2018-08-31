# Mainly adapted from OpenNMT.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var


class Attention(nn.Module):
    def __init__(self, opt, dim, attn_type="general", coverage=False, nn_feat_dim=None, sp_feat_dim=None):
        super(Attention, self).__init__()
        self.opt = opt
        self.dim = dim
        self.attn_type = attn_type

        if self.attn_type == 'general':
            self.W_h = nn.Linear(dim, dim, bias=True)
            self.W_s = nn.Linear(dim * 2, dim, bias=True)
            # self.W_prev_attn = nn.Linear(1, dim, bias=True)
        # elif self.attn_type == 'concat':
        #     self.linear_context = nn.Linear(dim * 2, dim)
        #     self.linear_bottle = nn.Linear(dim, 1)
        # elif self.attn_type == 'dot':
        #     pass
        else:
            raise NotImplementedError

        if coverage:
            self.W_coverage = nn.Linear(1, dim, bias=True)
        if sp_feat_dim is not None:
            self.sp = True
            _feat_num_activat = sum([1 if i else 0 for i in [opt.feat_word, opt.feat_ent, opt.feat_sent]])
            self.W_sp = nn.Linear(_feat_num_activat * sp_feat_dim, dim, bias=True)
        else:
            self.sp = False
        if nn_feat_dim is not None:
            self.nn = True
            self.W_nn = nn.Linear(nn_feat_dim, dim, bias=True)
        else:
            self.nn = False
        self.v = nn.Linear(dim, 1)

        # if coverage is not None:
        #     self.linear_coverage = nn.Linear(1, dim, bias=False)

        self.mask = None

    def masked_attention(self, e, mask):
        """Take softmax of e then apply enc_padding_mask and re-normalize"""
        max_e = torch.max(e, dim=1, keepdim=True)[0]
        e = e - max_e
        attn_dist = F.softmax(e)  # take softmax. shape (batch_size, attn_length)
        attn_dist = attn_dist * Var(mask.float(), requires_grad=False).cuda()  # apply mask
        masked_sums = torch.sum(attn_dist, dim=1, keepdim=True)  # shape (batch_size)
        masked_sums = masked_sums.expand_as(attn_dist)
        return attn_dist / masked_sums

    def score(self, h_i, s_t):
        """
        s_t (FloatTensor): batch x 1 x dim
        h_i (FloatTensor): batch x src_len x dim
        returns scores (FloatTensor): batch x 1 x src_len:
        raw attention scores for each src index
        """
        src_batch, src_len, src_dim = h_i.size()
        tgt_batch, tgt_len, tgt_dim = s_t.size()
        assert src_batch == tgt_batch
        assert src_dim == tgt_dim
        assert tgt_len == 1
        if self.attn_type == 'dot':
            h_s_ = h_i.transpose(1, 2)
            return torch.bmm(s_t, h_s_).squeeze()
        elif self.attn_type == 'general':
            h_s_ = self.linear_in(h_i)
            return torch.bmm(s_t, h_s_.transpose(1, 2)).squeeze()
        elif self.attn_type == 'concat':
            concat = torch.cat((h_i, s_t.expand_as(h_i)), dim=2)
            x = self.linear_context(concat)
            x = self.tanh(x)
            x = self.linear_bottle(x).squeeze()
            return x
        else:
            raise NotImplementedError

    def forward(self, current_state, context, context_mask, last_attn, coverage=None, feats=None):
        """

        :param current_state : (FloatTensor): batch x dim: decoder's rnn's output.
        :param context: (FloatTensor): batch x src_len x dim: src hidden states
        :param coverage: (FloatTensor): batch, src_len
        :return: attn_h, batch x dim; attn_vec, batch x context_len
        """
        batch_size, src_len, dim = context.size()
        batch_, dim_ = current_state[0].size()
        cat_current_state = torch.cat(current_state, dim=1)
        assert batch_size == batch_
        assert dim == dim_

        # sp_feat, nn_feat = feats

        # Compute coverage
        w_cov = 0
        if coverage is not None:
            batch_, src_len_ = coverage.size()
            assert batch_ == batch_size
            assert src_len_ == src_len
            # coverage = coverage.view(-1).unsqueeze(1)
            coverage = coverage + 0.001
            cov_sum = torch.sum(coverage, dim=1, keepdim=True)
            coverage = coverage / cov_sum
            coverage = coverage.unsqueeze(2)  # batch, src_len, 1
            w_cov = self.W_coverage(coverage)  # batch, src_len, dim

            cov_msk = torch.max(1 - coverage, 0)[0]

        # Compute s_t
        w_state = self.W_s(cat_current_state)  # batch_,  dim_
        w_state = w_state.unsqueeze(1)  # batch, 1, dim

        # Compute context h_i
        w_context = self.W_h(context)
        # batch_size, src_len, dim
        # last_attn = last_attn.unsqueeze(2)
        # w_prev_attn = self.W_prev_attn(last_attn)

        w_sp = 0
        if self.sp:
            feats = feats.view(batch_size, src_len, -1)
            w_sp = self.W_sp(feats)
            if coverage is not None:
                w_sp = w_sp * cov_msk

        activated = F.tanh(w_state + w_context + w_cov + w_sp)
        e = self.v(activated).squeeze(2)
        max_e = torch.max(e, dim=1, keepdim=True)[0]
        e = e - max_e

        attn_dist = self.masked_attention(e, context_mask)  # batch, seq
        exp_attn_dist = attn_dist.unsqueeze(2).expand_as(context)
        attn_h_weighted = torch.sum(exp_attn_dist * context, dim=1)

        # batch, dim

        # attn_h = self.linear_out(inp_contxt).view(batch_size, 1, dim)
        # if self.attn_type in ['general', 'dot']:
        #     attn_h = self.tanh(attn_h)
        # attn_h = attn_h.squeeze(1)
        # align_vec = align_vec.squeeze(1)

        return attn_h_weighted, attn_dist


if __name__ == "__main__":
    # Attention test
    import numpy as np

    dim = 5

    batch_size = 3
    src_len = 4
    input = np.arange(batch_size * dim).reshape((batch_size, 1, dim))
    input = Var(torch.FloatTensor(input))

    context = np.arange(batch_size * dim * src_len).reshape((batch_size, src_len, dim))
    context = Var(torch.FloatTensor(context))
    coverage = np.arange(batch_size * src_len).reshape((batch_size, src_len))
    coverage = Var(torch.FloatTensor(coverage))
    attn = Attention(dim, attn_type='general')

    mask = np.zeros((batch_size, src_len), dtype=int)
    mask[0, 2] = 1
    mask[1, 1] = 1
    mask[2, 2] = 1
    # Invalid position ==1!!!
    mask = torch.ByteTensor(mask)
    # mask = torch.ByteTensor(mask)
    attn.set_mask(mask)

    attn_h, align_vec = attn.forward(input, context, coverage)
    print(attn_h)
    print(align_vec)
    # print attn.score(Var(input), Var(context), Var(coverage))
    # attn = Attention(dim, attn_type='dot')
    # print attn.score(Var(input), Var(context))
    # attn = Attention(dim, attn_type='concat')
    # print attn.score(Var(input), Var(context))
