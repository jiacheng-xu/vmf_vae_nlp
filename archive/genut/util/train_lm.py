import logging
import math
import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable as Var

from archive.genut import msk_list_to_mat
from archive.genut import Trainer


class LMTrainer(Trainer):
    def __init__(self, opt, model, data):
        super().__init__(opt, model, data)
        # self.logger = Logger(opt.print_every, self.n_batch)
        # weight = torch.ones(opt.full_dict_size)
        # weight[0] = 0
        # assert 0 == opt.word_dict.fword2idx('<pad>')
        self.crit = nn.CrossEntropyLoss(size_average=True, reduce=True, ignore_index=0)
        self.crit_test = nn.CrossEntropyLoss(size_average=False, reduce=False, ignore_index=0)
        self.opt = opt
        self.model = model
        self.train_bag = data[0]
        self.test_bag = data[1]
        self.n_batch = len(self.train_bag)

        # parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        # self.optimizer = torch.optim.Adagrad(parameters, lr=opt.lr)  # TODO
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.lr, momentum=0.9)
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

        decoder_outputs_prob, decoder_outputs = self.model.forward(inp_var, inp_msk, inp_var, inp_msk, aux)

        valid_pos_mask = Var(msk_list_to_mat(inp_msk), requires_grad=False).view(target_len * batch_size, 1)
        if self.opt.use_cuda:
            valid_pos_mask = valid_pos_mask.cuda()

        # Compulsory NLL loss part
        pred_prob = decoder_outputs_prob.view(target_len * batch_size, -1)
        # print(inp_var)
        seq_first_inp_var = inp_var.transpose(1, 0).contiguous()
        gold_dist = seq_first_inp_var.view(target_len * batch_size)
        gold_dist = Var(gold_dist)
        if self.opt.use_cuda:
            gold_dist = gold_dist.cuda()
        loss = self.crit(pred_prob, gold_dist)

        loss.backward()

        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
        self.optimizer.step()

        return loss.data[0], math.exp(loss.data[0])

    def func_test(self, inp_var, inp_msk):
        target_len = inp_msk[0]
        batch_size = inp_var.size()[0]
        decoder_outputs_prob, decoder_outputs = self.model.forward(inp_var, inp_msk, tgt_var=inp_var, tgt_msk=inp_msk,
                                                                   aux=None)

        # Compulsory NLL loss part
        pred_prob = decoder_outputs_prob.view(target_len * batch_size, -1)
        seq_first_inp_var = inp_var.transpose(1, 0).contiguous()
        gold_dist = Var(seq_first_inp_var.view(target_len * batch_size))
        if self.opt.use_cuda:
            gold_dist = gold_dist.cuda()

        loss = self.crit_test(pred_prob, gold_dist)
        loss = torch.sum(loss)
        return loss.data[0], decoder_outputs

    def train_iters(self):
        """
        Training function called from main.py.
        :return:
        """
        for epo in range(self.opt.start_epo, self.opt.n_epo + 1):
            self.model.train()
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

                # inp_var = Var(inp_var)

                if self.opt.use_cuda:
                    # inp_var = [x.contiguous().cuda() for x in inp_var]
                    inp_var = inp_var.contiguous().cuda()

                nll, ppl = self.func_train(inp_var, inp_msk)

                if idx % self.opt.print_every == 0:
                    logging.info('NLL:%.2f \tPPL:%s' % (nll, str(ppl)))

                if idx % self.opt.save_every == 0:
                    ppl = self.evaluate()
                    os.chdir(self.opt.save_dir)

                    name_string = '%d_%.2f'.lower() % (epo, ppl)
                    logging.info("Saving in epo %s" % name_string)
                    torch.save(self.model.emb.state_dict(),
                               name_string + '_emb')
                    # torch.save(self.model.enc.state_dict(), name_string + '_enc')
                    torch.save(self.model.dec.state_dict(), name_string + '_dec')
                    torch.save(self.model.opt, name_string + '_opt')

                    os.chdir('..')

    def evaluate(self):
        self.model.eval()
        n_batch = len(self.test_bag)
        test_len = 0
        accumulated_ppl = 0
        for idx in range(n_batch):
            current_batch = self.test_bag[idx]
            inp_var = current_batch['txt']
            inp_mask = current_batch['txt_msk']
            batch_size = inp_var.size()[0]
            test_len += inp_mask[0] * batch_size
            nll, decoder_output = self.func_test(inp_var, inp_mask)
            accumulated_ppl += nll
        final_ppl = accumulated_ppl / test_len
        final_ppl = math.exp(final_ppl)
        logging.info('PPL: %f' % final_ppl)
        return final_ppl
