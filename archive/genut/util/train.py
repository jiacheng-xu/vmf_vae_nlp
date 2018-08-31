# from pythonrouge import summarizer
import copy
import datetime
import torch
import os
import numpy as np
from torch.autograd import Variable as Var


class Trainer():
    def __init__(self, opt, model, data):
        weight = torch.ones(opt.full_dict_size)
        weight[0] = 0
        assert 0 == opt.word_dict.fword2idx('<pad>')

        self.opt = opt
        self.model = model
        self.train_bag = data
        self.n_batch = len(self.train_bag)

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adagrad(parameters, lr=opt.lr)  # TODO
        # self.optimizer = torch.optim.SGD(parameters,lr=opt.lr)

        # self.mul_loss = opt.mul_loss
        # self.add_loss = opt.add_loss

        # dicts = [word_dict, pos_dict, ner_dict]
        self.word_dict = opt.word_dict

        self.clip = opt.clip
        # self.coverage = self.opt.coverage

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
        """
        Training function called from main.py.
        :return:
        """
        for epo in range(self.opt.start_epo, self.opt.n_epo + 1):
            self.logger.init_new_epo(epo)
            # Schedule
            self.opt.max_len_enc, self.opt.max_len_dec, self.cov_loss_weight = util.schedule(epo)

            batch_order = np.arange(self.n_batch)
            np.random.shuffle(batch_order)

            for idx, batch_idx in enumerate(batch_order):
                self.logger.init_new_batch(batch_idx)
                current_batch = self.train_bag[batch_idx]

                # current_batch = copy.deepcopy(tmp_cur_batch)

                inp_var = current_batch['txt']
                inp_msk = current_batch['txt_msk']

                # out_var = current_batch['cur_out_var']
                # out_mask = current_batch['cur_out_mask']
                # scatter_msk = current_batch['cur_scatter_mask'].cuda()
                # replacement = current_batch['replacement']
                # max_oov_len = len(replacement)
                # self.logger.set_oov(max_oov_len)

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

                # inp_var = [Var(x) for x in inp_var]
                inp_var = Var(inp_var)

                if self.opt.use_cuda:
                    # inp_var = [x.contiguous().cuda() for x in inp_var]
                    inp_var = inp_var.contiguous().cuda()

                self.func_train(inp_var, inp_msk, out_var, out_mask, features, feature_msks, max_oov_len,
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

    def saver(self):
        pass
