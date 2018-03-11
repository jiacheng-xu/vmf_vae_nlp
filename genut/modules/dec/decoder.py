import random
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable as Var


from genut.util.beam import Beam
from genut.modules.attention import Attention

class RNNDecoderBase(nn.Module):
    def __init__(self, opt, rnn_type='lstm', num_layers=1,
                 hidden_size=100, input_size=50, attn_type='dot', coverage=False,
                 copy=False, dropout=0.1, emb=None, full_dict_size=None, word_dict_size=None,
                 max_len_dec=100, beam=True):
        super(RNNDecoderBase, self).__init__()
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

    def forward(self, context, context_mask, state, tgt, tgt_mask, inp_var, feat, max_oov_len, scatter_mask,
                bigram_bunch,
                logger):
        """

        :param context: a PackedSequence obj
        :param context_mask: List, containing mask len. [300,250,200,....]
        :param state: tuple for LSTM.
        :param tgt:
        :param tgt_mask:
        :param inp_var:
        :param max_oov_len:
        :return:
        """
        if self.opt.mul_loss or self.opt.add_loss:
            bigram, bigram_msk, bigram_dicts = bigram_bunch
            bigram = Var(bigram, requires_grad=False).cuda()
            # bigram_msk = Var(bigram_msk, requires_grad=False).cuda()

            # window_msk = Var(window_msk, requires_grad=False).float().cuda().contiguous()

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
        Discount = Var(
            torch.zeros((tgt_len, batch_size, self.word_dict_size + max_oov_len))).cuda()  # Discount over output

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

        # if self._copy:
        #     logger.current_batch['p_gen'] = torch.mean(p_copys)
        return decoder_outputs_prob, decoder_outputs, Attns, Discount, loss_cov, p_copys

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

    def beam_decode(self, context, context_msk, state, inp_var, feat, max_oov, scatter_mask):
        """

        :param context: PackedSequence(seq_len * batch, hidden_size * num_directions)
        :param context_msk: [seq_len,..,.,...]
        :param state: tupple((batch, hidden_size * 2), ....)
        :param inp_var: seq_len, batch=1
        :return:
        """
        # context, context_mask_ = nn.utils.rnn.pad_packed_sequence(context)
        # assert context_msk == context_mask_
        contxt_len, batch_size, hdim = context.size()
        contxt_len_, batch_size_ = context_msk.size()
        context_len__, batch_size__ = inp_var.size()
        assert batch_size == 1 == batch_size_ == batch_size__
        assert context_len__ == contxt_len == contxt_len_
        # Till now, context: seq_len, batch_size, hidden_size (400)


        archived_hyp = []

        # Init for beam
        hyps = [Beam(opt=self.opt, tokens=[self.opt.sos],
                     log_probs=[0.0],
                     state=state, prev_attn=[torch.zeros(contxt_len)], p_gens=[],
                     coverage=torch.zeros((contxt_len)).cuda() if self._coverage else None
                     # zero vector of length attention_length
                     ) for _ in range(1)]

        # expanded_mask = util.mask_translator(context_msk, batch_first=True, is_var=True)  # TODO

        expanded_mask = context_msk.transpose(1, 0)

        expanded_context = context.expand((contxt_len, len(hyps), hdim))
        # expanded_mask = [context_msk[0] for t in range(len(hyps))]
        expanded_inp_var = inp_var.expand((contxt_len, len(hyps)))
        # self.set_mask(expanded_mask)
        if feat is not None:
            src_len, feat_dim = feat.size()
            expanded_feat = feat.unsqueeze(0)
        else:
            expanded_feat = None

        # if self._coverage:
        #     coverage = Var(torch.zeros((batch_size, contxt_len))).cuda()
        # else:
        #     coverage = None

        init = True

        steps = 0

        while steps < self.max_len_dec:

            # batch forward
            last_tokens = [h.latest_token for h in hyps]
            last_tokens = np.asarray(last_tokens, dtype=int).reshape(1, len(hyps))
            decoder_input = Var(torch.LongTensor(last_tokens), volatile=True).cuda()

            last_attn = [h.latest_attn for h in hyps]
            prev_attn = Var(torch.stack(last_attn)).cuda()
            # prev_attn = prev_attn.squeeze(1)
            if self._coverage:
                last_cov = [h.coverage for h in hyps]
                prev_cov = Var(torch.stack(last_cov))
            else:
                prev_cov = None
            states = [h.state for h in hyps]
            inp_h, inp_c = torch.FloatTensor(len(hyps), hdim), torch.FloatTensor(len(hyps), hdim)
            for idx, s in enumerate(states):
                inp_h[idx] = s[0].data
                inp_c[idx] = s[1].data  # TODO
            states = (Var(inp_h, volatile=True).cuda(), Var(inp_c, volatile=True).cuda())

            # input, context, prev_state, coverage = None, inp_var = None
            states, prob_final, coverage, attn, p_gen = self._run_forward_one(decoder_input, expanded_context,
                                                                              expanded_mask, expanded_feat, states,
                                                                              prev_attn,
                                                                              prev_cov,
                                                                              expanded_inp_var, max_oov, scatter_mask)
            # input, context, context_mask, feats, prev_state, prev_attn, coverage=None, inp_var=None,max_oov_len=None,scatter_mask=None
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                coverage = coverage.data

            all_hyps = []
            for seed in range(len(hyps)):
                prob = prob_final[seed]  # 50381
                inp_h = states[0][seed]
                inp_c = states[1][seed]
                topv, topi = prob.data.topk(self.beam_size)
                this_attn = attn[seed].data
                this_p_gen = p_gen[seed].data
                if self._coverage:
                    this_cov = coverage[seed]
                else:
                    this_cov = None
                for b in range(self.beam_size):
                    three_gram = None
                    if len(hyps[seed].tokens) > 2:
                        three_gram = "%d_%d_%d" % (hyps[seed].tokens[-2], hyps[seed].tokens[-1], topi[b])

                    bi_gram = None
                    if len(hyps[seed].tokens) > 1:
                        bi_gram = "%d_%d" % (hyps[seed].tokens[-1], topi[b])

                    new_hyp = hyps[seed].extend(self.opt, topi[b], log_prob=topv[b],
                                                state=(inp_h, inp_c), coverage=this_cov, bi_gram=bi_gram,
                                                three_gram=three_gram,
                                                prev_attn=this_attn, p_gen=this_p_gen)
                    # new_hyp.avid_repeatition()
                    # if repeated:
                    #     print("Repeated")
                    all_hyps.append(new_hyp)

            # print("Step: %d"%(steps))
            hyps = []
            # print('--------')
            for h in util.sort_hyps(all_hyps):
                # print(h.avg_log_prob())
                # print(h.log_probs)
                # print(h.tokens)
                if h.latest_token == self.opt.eos and len(h.tokens) > self.opt.min_len_dec:
                    archived_hyp.append(h)
                elif h.latest_token == self.opt.eos:
                    h.tokens = h.tokens[:-1]
                    h.log_probs = h.log_probs[:-1]
                    h.prev_attn = h.prev_attn[:-1]
                    h.p_gens = h.p_gens[:-1]
                    hyps.append(h)
                else:
                    hyps.append(h)
                if len(hyps) == self.beam_size:
                    break

            if len(archived_hyp) > 1:
                archived_hyp = util.sort_hyps(archived_hyp)
                if len(archived_hyp) > 1:
                    archived_hyp = [archived_hyp[0]]

            steps += 1
            if len(hyps) == 0:
                break

            if init:
                init = False
                scatter_mask = scatter_mask.expand(len(hyps), contxt_len, contxt_len)
                expanded_context = context.expand((contxt_len, len(hyps), hdim))
                # expanded_dig_mask = [context_msk[0] for t in range(len(hyps))]
                # expanded_mask = util.mask_translator(expanded_dig_mask, batch_first=True, is_var=True)
                expanded_mask = context_msk.transpose(1, 0).repeat(len(hyps), 1)
                expanded_inp_var = inp_var.expand((contxt_len, len(hyps)))
                if feat is not None:
                    expanded_feat = expanded_feat.expand((len(hyps), contxt_len_, feat_dim)).contiguous()
                else:
                    expanded_feat = None
        best = None
        best_alive = None
        best_archived = None
        if len(hyps) > 0:
            best_alive = hyps[0]
        if len(archived_hyp) > 0:
            best_archived = archived_hyp[0]

        if best_archived is not None and best_alive is not None:
            best = best_archived if best_archived.avg_log_prob() > best_alive.avg_log_prob() else best_alive
        elif best_archived is None:
            best = best_alive
        elif best_alive is None:
            best = best_archived
        else:
            raise ("WTF")

        return best.tokens[1:], best.prev_attn[1:], best.p_gens
        # return decoder_outputs_prob, __ , decoder_outputs, attn_bag
