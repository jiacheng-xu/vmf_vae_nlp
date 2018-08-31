import copy
import datetime
import torch
import os

import util
from CONSTANTS import *
from util import *
from util import Logger

# import GPUtil
stop_words = [',', '.', 'to', 'the', '<\\s>', '<s>', 'a', 'of', 'he', 'is', 'was', 'on', 'have', 'has', 'be', '`', '\'',
              'it',
              'in']


class Trainer(object):
    def __init__(self, opt, model, dicts, data):
        weight = torch.ones(opt.full_dict_size)
        weight[PAD] = 0
        assert PAD == dicts[0].fword2idx('<pad>')
        # if opt.mul_loss:
        #     self.crit = torch.nn.NLLLoss(size_average=False, ignore_index=PAD, reduce=False)
        # else:
        #     self.crit = torch.nn.NLLLoss(size_average=True, ignore_index=PAD)
        self.opt = opt
        self.model = model
        self.train_bag = data
        self.n_batch = len(self.train_bag)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adagrad(parameters, lr=opt.lr)  # TODO
        # self.optimizer = torch.optim.SGD(parameters,lr=opt.lr)

        self.mul_loss = opt.mul_loss
        self.add_loss = opt.add_loss

        # dicts = [word_dict, pos_dict, ner_dict]
        self.word_dict = dicts[0]
        self.pos_dict = dicts[1]
        self.ner_dict = dicts[2]

        self.clip = opt.clip
        # self.val_bag = val_data
        self.bool_test = False
        self.coverage = self.opt.coverage
        self.logger = Logger(opt.print_every, self.n_batch)
        self.histo = None

    def assert_special_chars(self):
        return None
        # print self.word_dict.word2idx['<eos>']
        assert self.word_dict.word2idx['<eos>'] == EOS
        assert self.word_dict.word2idx['<pad>'] == PAD
        assert self.word_dict.word2idx['<sos>'] == SOS
        assert self.word_dict.word2idx['<unk>'] == UNK

    def weighted_loss(self, decoder_outputs_prob, decoder_outputs, tgt_var):
        """

        :param decoder_outputs_prob: seq, batch, dict_size
        :param decoder_outputs: seq, batch
        :param tgt_var: seq, batch
        :return:
        """
        gram = [2, 3]
        weight = [0.1, 0.01]
        seq_len, batch_size = tgt_var.size()[0], tgt_var.size()[1]
        seq_len__, batch_size__ = decoder_outputs.size()[0], decoder_outputs.size()[1]
        seq_len_, batch_size_, dict_size = decoder_outputs_prob.size()[0], decoder_outputs_prob.size()[1] \
            , decoder_outputs_prob.size()[2]
        assert seq_len == seq_len_ == seq_len__
        assert batch_size == batch_size_ == batch_size__

        gold_n_grams = n_gram_list(gram, tgt_var)

        weight_mat = torch.ones(decoder_outputs.size())
        for b in range(batch_size):
            pred_seq = decoder_outputs[:, b]
            for t in range(seq_len_):
                for j, g in enumerate(gram):
                    if t - g + 1 >= 0:
                        feat = ''
                        for x in range(t - g + 1, t + 1, 1):
                            feat += str(pred_seq[x]) + '_'
                        if feat in gold_n_grams[b][j]:
                            for x in range(t - g + 1, t + 1, 1):
                                weight_mat[x, b] *= weight[j]
        weight_mat = weight_mat.view(seq_len * batch_size, -1)
        # print(torch.mean(weight_mat))
        loss = self.crit(decoder_outputs_prob.view(seq_len * batch_size, -1),
                         Var(tgt_var).view(seq_len * batch_size).cuda())
        original_loss = torch.sum(loss)
        cu_weights = Var(weight_mat.squeeze()).cuda()
        loss = torch.sum(torch.mul(loss, cu_weights))
        # cu_weights = Var(weight_mat.transpose()).cuda()
        # loss = loss * cu_weights
        return loss, original_loss

    def train_iters(self):

        for epo in range(self.opt.start_epo, self.opt.n_epo + 1):
            self.logger.init_new_epo(epo)
            # Schedule
            self.opt.max_len_enc, self.opt.max_len_dec, self.cov_loss_weight = util.schedule(epo)
            # if self.opt.max_len_enc == self.histo:
            #     need_to_recompute = False
            # else:
            #     need_to_recompute = True
            #     self.histo = self.opt.max_len_enc

            # self.train_bag = self.model.feat.update_msks(self.opt.max_len_enc, self.opt.max_len_dec, self.train_bag,
            #                                              self.ner_dict)

            # if self.opt.feat_sp or self.opt.feat_nn:
            #     self.train_bag = self.model.feat.sp.extract_feat(self.opt, self.train_bag,
            #                                                      [self.pos_dict, self.ner_dict])

            batch_order = np.arange(self.n_batch)
            np.random.shuffle(batch_order)

            for idx, batch_idx in enumerate(batch_order):
                self.logger.init_new_batch(batch_idx)
                tmp_cur_batch = self.train_bag[batch_idx]

                current_batch = copy.deepcopy(tmp_cur_batch)
                # if need_to_recompute:
                current_batch = self.model.feat.update_msks_batch(
                    self.opt, self.opt.mode, self.opt.max_len_enc, self.opt.max_len_dec, current_batch, self.pos_dict,
                    self.ner_dict)

                inp_var = current_batch['cur_inp_var']
                inp_mask = current_batch['cur_inp_mask']
                out_var = current_batch['cur_out_var']
                out_mask = current_batch['cur_out_mask']
                scatter_msk = current_batch['cur_scatter_mask'].cuda()
                replacement = current_batch['replacement']
                max_oov_len = len(replacement)
                self.logger.set_oov(max_oov_len)

                if self.opt.feat_word or self.opt.feat_ent or self.opt.feat_sent:
                    features = [current_batch['word_feat'], current_batch['ent_feat'], current_batch['sent_feat']]
                    feature_msks = [current_batch['cur_word_msk'], current_batch['cur_ent_msk'],
                                    current_batch['cur_sent_msk']]
                else:
                    features = None
                    feature_msks = None
                if self.opt.mul_loss or self.opt.add_loss:
                    bigram = current_batch['bigram']
                    # print(torch.sum(bigram))
                    bigram_msk = current_batch['bigram_msk']
                    bigram_dict = current_batch['bigram_dict']
                    bigram_bunch = [bigram, bigram_msk, bigram_dict]
                    # print(torch.sum(window_msk))
                else:
                    bigram_bunch = None
                # inp_var = util.truncate_mat(self.opt.max_len_enc, inp_var)
                # out_var = util.truncate_mat(self.opt.max_len_dec, out_var)

                # Sparse Feature preload
                # sparse_feat_indicator_mat = self.model.feat_sp.generate_feature_indicator(inp_var, ori_txt,
                #                                                                           self.ner_dict)

                # # Need to generate Mask for both txt and abs
                # inp_mask = torch.gt(inp_var[0], 0)
                # out_mask = torch.gt(out_var[0], 0)
                # # Need to fix Attention Supervision since some of the the raw text are lost after truncation
                # neg_mask = torch.ge(out_var[1], self.opt.max_len_enc)
                # out_var[1] = out_var[1].masked_fill_(neg_mask, -1)
                # # out_var[1] = out_var[1] * neg_mask
                #
                # scatter_mask = util.prepare_scatter_map(inp_var[0])

                inp_var = [Var(x) for x in inp_var]

                if self.opt.use_cuda:
                    inp_var = [x.contiguous().cuda() for x in inp_var]

                self.func_train(inp_var, inp_mask, out_var, out_mask, features, feature_msks, max_oov_len,
                                scatter_msk, bigram_bunch)

                if idx % self.opt.save_every == 0:
                    #######
                    # Saving
                    # End of Epo
                    print_loss_avg = sum(self.logger.lm.history_loss['NLL']) / len(self.logger.lm.history_loss['ALL'])
                    # print_loss_avg = sum(self.logger.current_epo['loss']) / self.logger.current_epo['count']
                    os.chdir(self.opt.save_dir)

                    name_string = '%d_%.3f_%s_Cop%s_Cov%s_%dx%s_%s%s_E%d_D%d_DL%s_%01.1f_SL%s_%01.1f_Attn%s_%01.1f_Feat%s'.lower() % (
                        epo, print_loss_avg, str(self.opt.enc), str(self.opt.copy), str(self.opt.coverage),
                        self.model.opt.full_dict_size, datetime.datetime.now().strftime("%B%d%I%M"),
                        self.opt.data_path.split('/')[-1],
                        self.opt.name, self.opt.max_len_enc, self.opt.max_len_dec, str(self.opt.mul_loss),
                        self.opt.lw_bgdyn,
                        str(self.opt.add_loss), self.opt.lw_bgsta, str(self.opt.attn_sup), self.opt.lw_attn,
                        str(self.opt.feat_sp)
                    )
                    print(name_string)
                    torch.save(self.model.emb.state_dict(),
                               name_string + '_emb')
                    torch.save(self.model.feat, name_string + '_feat')
                    torch.save(self.model.enc.state_dict(), name_string + '_enc')
                    torch.save(self.model.dec.state_dict(), name_string + '_dec')
                    torch.save(self.model.opt, name_string + '_opt')

                    os.chdir('..')
            # End Saving
            ########

        print('\n')

    def func_train(self, inp_var, inp_msk, out_var, tgt_msk, features, feature_msks, max_oov_len, scatter_mask,
                   bigram_bunch):
        self.optimizer.zero_grad()  # clear grad

        tgt_var, attn_sup = out_var

        batch_size = inp_var[0].size()[1]
        batch_size_ = tgt_var.size()[1]
        assert batch_size == batch_size_ == inp_var[2].size()[1] == inp_var[1].size()[1]

        target_len = tgt_var.size()[0]
        src_len = inp_var[0].size()[0]

        self.logger.current_batch['valid_pos'] = torch.sum(tgt_msk)

        decoder_outputs_prob, decoder_outputs, attns, discount, loss_cov, p_copys = self.model.train_forward(inp_var,
                                                                                                             tgt_var,
                                                                                                             inp_msk,
                                                                                                             tgt_msk,
                                                                                                             features,
                                                                                                             feature_msks,
                                                                                                             max_oov_len,
                                                                                                             scatter_mask,
                                                                                                             bigram_bunch,
                                                                                                             self.logger)
        tgt_padding_mask = Var(tgt_msk.float(), requires_grad=False).cuda().view(target_len * batch_size, 1)

        # print(decoder_outputs_prob, decoder_outputs, attns, discount, loss_cov)
        # decoder_outputs_prob: tgt,batch,full_vocab
        # decoder_outputs: tgt, batch
        # attns: tgt, batch, src_len
        # discount: tgt, batch, full_vocab
        # loss_cov: tgt, batch
        # p_copys: batch, tgt   Value only

        if self.opt.copy:
            self.logger.lm.add_LossItem(LossItem(name='pgen', node=torch.mean(p_copys), weight=0))

        if self.coverage:
            # loss_cov: tgt, batch
            flat_loss_cov = loss_cov.view(target_len * batch_size, 1)
            val_loss_cov = torch.mean(tgt_padding_mask * flat_loss_cov)
            self.logger.lm.add_LossItem(LossItem(name='cov', node=val_loss_cov, weight=self.opt.lw_cov))

        if self.opt.attn_sup:
            # attn_sup: tgt_len, batch LongTensor
            # attns: tgt_len, batch, src_len floatTensor
            tgt_sz, batch_sz = attn_sup.size()
            tgt_sz_, batch_sz_, src_len_ = attns.size()
            assert batch_sz == batch_size_
            flat_attn_sup = Var(attn_sup.view(-1)).cuda()
            flat_attns = attns.view(-1, src_len_)

            valid_attn_msk = torch.gt(flat_attn_sup, 0)
            valid_tgt = torch.masked_select(flat_attn_sup, valid_attn_msk)

            extded_valid_attn_msk = valid_attn_msk.view(-1, 1).expand_as(flat_attns)
            valid_pred = torch.masked_select(flat_attns, extded_valid_attn_msk)

            valid_pred = valid_pred.view(-1, src_len_)
            loss_attn = -torch.log(torch.gather(valid_pred, 1, valid_tgt.unsqueeze(1)))
            loss_attn = torch.mean(loss_attn)
            self.logger.lm.add_LossItem(LossItem(name='attn', node=loss_attn, weight=self.opt.lw_attn))
            # loss_attn = 1 - torch.mean(torch.gather(valid_pred, 0, valid_tgt))

        # Compulsory NLL loss part
        pred_prob = decoder_outputs_prob.view(target_len * batch_size, -1)
        gold_dist = Var(tgt_var).view(target_len * batch_size, 1).cuda()
        losses = -torch.gather(pred_prob, 1, gold_dist)
        losses = losses * tgt_padding_mask
        # Then for -inf mask. a word neither exists in vocab nor exists in source article will be
        # -inf.
        inf_mask = losses.le(10000)
        nll_loss = torch.masked_select(losses, inf_mask)
        nll_loss = torch.mean(nll_loss)
        self.logger.lm.add_LossItem(LossItem(name='NLL', node=nll_loss, weight=self.opt.lw_nll))

        if self.mul_loss:
            # discount: tgt, batch, full_vocab

            # discount = torch.min(discount, torch.ones_like(discount))  # should naturally <1
            # discount = -torch.masked_select(discount, torch.gt(discount, 0))
            discount = 2 - torch.sum(discount) / 200

            # discount = torch.mean(discount)
            self.logger.lm.add_LossItem(LossItem(name='BGdyn', node=discount, weight=self.opt.lw_bgdyn))
        elif self.add_loss:
            # print(discount)
            _msk = torch.gt(discount, 0).view(target_len * batch_size, -1).float()

            # x = torch.masked_select(losses, _msk)
            discount = 1 - torch.sum(losses * _msk) / 500
            # print(x)

            # reward_msk = 1 - torch.gt(discount, 0).float().view(target_len * batch_size, -1) * 99./100.
            # sta_weighted_loss = torch.mean(losses * reward_msk)
            self.logger.lm.add_LossItem(LossItem(name='BGsta', node=discount, weight=self.opt.lw_bgsta))

        loss = self.logger.lm.compute()
        loss.backward()
        # print('update')
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
        self.optimizer.step()
        self.logger.batch_end()
