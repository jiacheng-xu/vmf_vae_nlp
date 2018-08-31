import numpy as np
import torch
import torch.optim
from torch.autograd import Variable as Var

from archive.genut import Beam


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
