# Load nvdm or nvrnn, check how Gauss or vMF distributes
"Dataptb_Distnor_Modelnvrnn_Emb400_Hid400_lat200_lr0.001_drop0.2"

import os
import random
import time

import numpy
import scipy
import torch

from NVLL.data.lm import DataLM
from NVLL.framework.train_eval_nvrnn import Runner
from NVLL.model.nvrnn import RNNVAE
from NVLL.util.util import GVar, swap_by_batch, replace_by_batch


class PlayNVRNN():
    def __init__(self, load_path, load_name, data_path, swap, replace, mix_unk):

        self.args = self.load_args(load_path, load_name)
        print(swap, replace, mix_unk)
        if swap is not None:
            self.args.swap = swap
        if replace is not None:
            self.args.replace = replace
        if mix_unk is not None:
            self.args.mix_unk = mix_unk
        self.data = self.load_data(data_path)
        self.model = self.load_model(load_path, load_name)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.detail_crit = torch.nn.CrossEntropyLoss(ignore_index=0, reduce=False)

    def load_data(self, data_path):
        data = DataLM(data_path, self.args.batch_size, self.args.eval_batch_size)
        return data

    def load_args(self, path, name):
        from NVLL.argparser import parse_arg

        with open(os.path.join(path, name + '.args'), 'rb') as f:
            args = torch.load(f)
        return args

    def load_model(self, path, name):
        model = RNNVAE(self.args, self.args.enc_type, len(self.data.dictionary), self.args.emsize,
                       self.args.nhid, self.args.lat_dim, self.args.nlayers,
                       dropout=self.args.dropout, tie_weights=self.args.tied,
                       input_z=self.args.input_z, mix_unk=self.args.mix_unk,
                       condition=(self.args.cd_bit or self.args.cd_bow),
                       input_cd_bow=self.args.cd_bow, input_cd_bit=self.args.cd_bit)
        model.load_state_dict(torch.load(os.path.join(path, name + '.model')))
        model = model.cuda()
        return model

    def eva(self):
        # Load the best saved model.
        cur_loss, cur_kl, test_loss = self.evaluate(self.args, self.model,
                                                    self.data.test)
        Runner.log_eval(None, 0, cur_loss, cur_kl, test_loss, True)
        return cur_loss, cur_kl, test_loss

    def evaluate(self, args, model, dev_batches):

        # Turn on training mode which enables dropout.
        model.eval()
        model.FLAG_train = False

        acc_loss = 0
        acc_kl_loss = 0
        acc_aux_loss = 0
        acc_avg_cos = 0
        acc_avg_norm = 0

        batch_cnt = 0
        all_cnt = 0
        cnt = 0
        start_time = time.time()

        for idx, batch in enumerate(dev_batches):
            feed = self.data.get_feed(batch)
            target = GVar(batch)
            seq_len, batch_sz = batch.size()

            if self.args.swap > 0.00001:
                feed = swap_by_batch(feed, self.args.swap)
            if self.args.replace > 0.00001:
                feed = replace_by_batch(feed, self.args.replace, self.model.ntoken)

            recon_loss, kld, aux_loss, tup, vecs = model(feed, target)

            acc_loss += recon_loss.data * seq_len * batch_sz
            acc_kl_loss += torch.sum(kld).data
            acc_aux_loss += torch.sum(aux_loss).data
            acc_avg_cos += tup['avg_cos'].data
            acc_avg_norm += tup['avg_norm'].data

            cnt += 1
            batch_cnt += batch_sz
            all_cnt += batch_sz * seq_len

        cur_loss = acc_loss[0] / all_cnt
        cur_kl = acc_kl_loss[0] / all_cnt
        cur_aux_loss = acc_aux_loss[0] / all_cnt
        cur_avg_cos = acc_avg_cos[0] / cnt
        cur_avg_norm = acc_avg_norm[0] / cnt
        cur_real_loss = cur_loss + cur_kl

        # Runner.log_eval(print_ppl)
        # print('loss {:5.2f} | KL {:5.2f} | ppl {:8.2f}'.format(            cur_loss, cur_kl, math.exp(print_ppl)))
        return cur_loss, cur_kl, cur_real_loss

    def play_eval(self, args, model, train_batches, epo, epo_start_time, glob_iter):
        # reveal the relation between latent space and length and loss
        # reveal the distribution of latent space
        model.eval()
        model.FLAG_train = False
        start_time = time.time()

        acc_loss = 0
        acc_kl_loss = 0
        acc_aux_loss = 0
        acc_avg_cos = 0
        acc_avg_norm = 0

        batch_cnt = 0
        all_cnt = 0
        cnt = 0
        random.shuffle(train_batches)

        if self.args.dist == 'nor':
            vs = visual_gauss(self.data.dictionary)
        elif self.args.dist == 'vmf':
            vs = visual_vmf(self.data.dictionary)

        for idx, batch in enumerate(train_batches):
            seq_len, batch_sz = batch.size()
            feed = self.data.get_feed(batch)

            glob_iter += 1

            target = GVar(batch)

            recon_loss, kld, aux_loss, tup, vecs = model(feed, target)

            acc_loss += recon_loss.data * seq_len * batch_sz
            acc_kl_loss += torch.sum(kld).data
            acc_aux_loss += torch.sum(aux_loss).data
            acc_avg_cos += tup['avg_cos'].data
            acc_avg_norm += tup['avg_norm'].data

            cnt += 1
            batch_cnt += batch_sz
            all_cnt += batch_sz * seq_len

            vs.add_batch(target.data, tup, kld.data)

        cur_loss = acc_loss[0] / all_cnt
        cur_kl = acc_kl_loss[0] / all_cnt
        cur_aux_loss = acc_aux_loss[0] / all_cnt
        cur_avg_cos = acc_avg_cos[0] / cnt
        cur_avg_norm = acc_avg_norm[0] / cnt
        cur_real_loss = cur_loss + cur_kl
        Runner.log_instant(None, self.args, glob_iter, epo, start_time, cur_avg_cos, cur_avg_norm,
                           cur_loss
                           , cur_kl, cur_aux_loss,
                           cur_real_loss)
        vs.write_log()


class visual_gauss():
    def __init__(self, d):
        self.logs = []
        self.dict = d

    def add_batch(self, target, tup, kld, loss):
        seq_len, batch_sz = loss.size()
        _seq_len, _batch_sz = target.size()
        __batch = kld.size()[0]
        assert seq_len == _seq_len
        assert batch_sz == _batch_sz == __batch
        mean = tup['mean']
        logvar = tup['logvar']
        # print(target.size())
        # print(batch_sz)
        for b in range(batch_sz):
            this_target = target[:, b]
            this_mean = mean[b]
            this_logvar = logvar[b]
            this_kld = kld[b]
            this_loss = loss[:, b]
            self.add_single(this_target, this_mean, this_logvar,
                            this_kld, this_loss)

    def add_single(self, target, mean, logvar, kld, loss):
        norm_mean = torch.norm(mean).data[0]
        norm_var = torch.norm(torch.exp(logvar)).data[0]
        length = len(target)
        seq = ''
        for t in target:
            seq += self.dict.idx2word[t] + '_'

        self.logs.append("{}\t{}\t{}\t{}\t{}\t{}".format(norm_mean, norm_var, kld, torch.mean(loss)
                                                         , length, seq))

    def write_log(self):
        with open('vslog.txt', 'w') as f:
            f.write('\n'.join(self.logs))


class visual_vmf():
    def __init__(self, d):
        self.logs = []
        self.dict = d

    def add_batch(self, target, tup, kld):
        _seq_len, _batch_sz = target.size()
        # __batch = kld.size()[0]
        mu = tup['mu']
        # print(target.size())
        # print(batch_sz)
        for b in range(_batch_sz):
            this_target = target[:, b]
            this_mu = mu[b]
            self.add_single(this_target, this_mu)

    def add_single(self, target, mu):
        thismu = mu.data
        length = len(target)
        seq = ''
        for t in target:
            seq += self.dict.idx2word[t] + '_'

        # self.logs.append("{}\t{}\t{}\t{}\t{}\t{}".format(norm_mean,kld,torch.mean(loss)
        #                                      ,length, seq))
        tmp = []
        for i in thismu:
            tmp.append(str(i))
        s = '\t'.join(tmp)
        self.logs.append(s)

    def write_log(self):
        with open('vh.txt', 'w') as f:
            f.write('\n'.join(self.logs))
        # with open('vu.txt', 'w') as f:
        #     f.write('\n'.join(self.logs))


def query(word):
    with open('/home/jcxu/vae_txt/data/ptb/test.txt', 'r') as f:
        lines = f.read().splitlines()
    bag = []
    for l in lines:
        if word in l:
            bag.append(l)
    with open('/home/jcxu/vae_txt/data/ptb/test_' + word + '.txt', 'w') as f:
        f.write('\n'.join(bag))


import scipy.spatial.distance as ds


def compute_cos(files):
    bags = []
    for fname in files:
        with open(fname, 'r') as fd:
            lines = fd.read().splitlines()
        bag = []
        for l in lines:
            nums = []
            tabs = l.split('\t')
            for t in tabs:
                nums.append(float(t))
            x = torch.FloatTensor(numpy.asarray(nums))
            bag.append(x)
        bags.append(bag)

    def _mean_of_bag(bag):
        x = 0
        for b in range(len(bag)):
            x += bag[b]
        tmp = x / len(bag)
        # print('avg of bag {}'.format(tmp))
        return tmp

    def comp_cos(a, b):
        return (torch.sum(a * b) / (torch.norm(a) * torch.norm(b)))

    A = bags[0]  # h
    B = bags[1]  # j

    print(comp_cos(_mean_of_bag(A), _mean_of_bag(B)))
    print('-' * 50)
    arec = []

    for idx, aa in enumerate(A):
        for jdx in range(idx, len(A)):
            print('{}\t{}\t{}'.format(idx, jdx, comp_cos(aa, A[jdx])))
            arec.append(comp_cos(aa, A[jdx]))
    print(sum(arec) / float(len(arec)))
    print('-' * 50)
    brec = []
    for idx, aa in enumerate(B):
        for jdx in range(idx, len(B)):
            print("{}\t{}\t{}".format(idx, jdx, comp_cos(aa, B[jdx])))
            brec.append(comp_cos(aa, B[jdx]))
    print(sum(brec) / float(len(brec)))


if __name__ == '__main__':
    # bag = []
    # for swap in [0.,0.25,0.5,1]:
    #     for replace in [0.,0.25,0.5,1]:
    #         for unk in [0.,0.25,0.5,1]:
    #
    #
    #             player = PlayNVRNN('/backup2/jcxu/exp-nvrnn',
    #                                'Dataptb_Distnor_Modelnvrnn_Emb100_Hid400_lat32_lr0.1_drop0.7_kappa16.0_auxw0.0_normfFalse_nlay1_mixunk1.0_inpzTrue'
    #                                , '/home/jcxu/vae_txt/data/ptb',swap=swap,replace=replace,mix_unk=unk)
    #             cur_loss, cur_kl, test_loss = player.eva()
    #             s = '{}\t{}\t{}\t{}\t{}\t{}'.format(swap, replace, unk, cur_loss,cur_kl,cur_loss)
    #             bag.append(s)
    #             print(bag)
    # for b in bag:
    #     print(b)
    player = PlayNVRNN('/backup2/jcxu/exp-nvrnn',
                       'Dataptb_Distvmf_Modelnvrnn_Emb100_Hid800_lat32_lr10.0_drop0.5_kappa64.0_auxw0.01_normfFalse_nlay1_mixunk1.0_inpzTrue'
                       , '/home/jcxu/vae_txt/data/ptb', swap=0, replace=0, mix_unk=1)
    cur_loss, cur_kl, test_loss = player.eva()
    # player.play_eval(player.args, player.model, player.data.demo_h, 0, 0, 0)

    # os.chdir('/home/jcxu/vae_txt/NVLL/framework')
    # compute_cos(['vu.txt', 've.txt'])
