import datetime
import os

import numpy as np
import torch
from torch.autograd import Variable as Var

from genut.util.helper import msk_list_to_mat
from genut.util.train import Trainer


class LMTrainer(Trainer):
    def __init__(self, opt, model, data):
        super().__init__(opt, model, data)
        # self.logger = Logger(opt.print_every, self.n_batch)
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

    def func_train(self, inp_var, inp_msk):
        self.optimizer.zero_grad()  # clear grad
        aux = {}
        batch_size = inp_var.size()[0]
        batch_size_ = len(inp_msk)
        assert batch_size == batch_size_

        target_len = inp_msk[0]

        # self.logger.current_batch['valid_pos'] = torch.sum(tgt_msk)

        decoder_outputs_prob, decoder_outputs = self.model.forward(inp_var, inp_msk, inp_var, inp_msk, aux)

        valid_pos_mask = Var(msk_list_to_mat(inp_msk), requires_grad=False).view(target_len * batch_size, 1)
        if self.opt.use_cuda:
            valid_pos_mask = valid_pos_mask.cuda()

        # Compulsory NLL loss part
        pred_prob = decoder_outputs_prob.view(target_len * batch_size, -1)
        # print(inp_var)
        seq_first_inp_var = inp_var.transpose(1, 0).contiguous()
        gold_dist = seq_first_inp_var.view(target_len * batch_size, 1)
        if self.opt.use_cuda:
            gold_dist = gold_dist.cuda()

        losses = -torch.gather(pred_prob, 1, gold_dist)
        # print(valid_pos_mask.size())
        # print(losses.size())
        losses = losses * valid_pos_mask
        # Then for -inf mask. a word neither exists in vocab nor exists in source article will be
        # -inf.
        # inf_mask = losses.le(10000)
        # nll_loss = torch.masked_select(losses, inf_mask)
        nll_loss = torch.mean(losses)

        nll_loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
        self.optimizer.step()

    def train_iters(self):
        """
        Training function called from main.py.
        :return:
        """
        for epo in range(self.opt.start_epo, self.opt.n_epo + 1):

            batch_order = np.arange(self.n_batch)
            np.random.shuffle(batch_order)

            for idx, batch_idx in enumerate(batch_order):
                # self.logger.init_new_batch(batch_idx)
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

                inp_var = Var(inp_var)

                if self.opt.use_cuda:
                    # inp_var = [x.contiguous().cuda() for x in inp_var]
                    inp_var = inp_var.contiguous().cuda()

                self.func_train(inp_var, inp_msk)

                if idx % self.opt.save_every == 0:
                    #######
                    # Saving
                    # End of Epo
                    # print_loss_avg = sum(self.logger.lm.history_loss['NLL']) / len(self.logger.lm.history_loss['ALL'])
                    # print_loss_avg = sum(self.logger.current_epo['loss']) / self.logger.current_epo['count']
                    os.chdir(self.opt.save_dir)

                    name_string = '%d_%.3f_%s_Cop%s_Cov%s_%dx%s_%s%s_E%d_D%d_DL%s_%01.1f_SL%s_%01.1f_Attn%s_%01.1f_Feat%s'.lower() % (
                        epo, print_loss_avg,
                        self.model.opt.full_dict_size, datetime.datetime.now().strftime("%B%d%I%M")
                    )
                    print(name_string)
                    torch.save(self.model.emb.state_dict(),
                               name_string + '_emb')
                    torch.save(self.model.enc.state_dict(), name_string + '_enc')
                    torch.save(self.model.dec.state_dict(), name_string + '_dec')
                    torch.save(self.model.opt, name_string + '_opt')

                    os.chdir('..')

    @staticmethod
    def calc_ppl(inp, inp_msk):
        pass