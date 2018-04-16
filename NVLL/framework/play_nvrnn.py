# Load nvdm or nvrnn, check how Gauss or vMF distributes
"Dataptb_Distnor_Modelnvrnn_Emb400_Hid400_lat200_lr0.001_drop0.2"

import os
import random
import time

import numpy
import scipy
import torch

from  NVLL.data.lm import DataLM
from NVLL.framework.run_nvrnn import Runner
from NVLL.model.nvrnn import RNNVAE
from NVLL.util.util import GVar


class PlayNVRNN():
    def __init__(self, load_path, load_name, data_path):
        self.args = self.load_args(load_path, load_name)
        self.data = self.load_data(data_path)
        self.model = self.load_model(load_path, load_name)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.detail_crit = torch.nn.CrossEntropyLoss(ignore_index=0, reduce=False)

    def load_data(self, data_path):
        data = DataLM(data_path, self.args.batch_size, self.args.eval_batch_size)
        return data

    def load_args(self, path, name):
        with open(os.path.join(path, name + '.args'), 'rb') as f:
            args = torch.load(f)
        return args

    def load_model(self, path, name):
        model = RNNVAE(self.args, self.args.enc_type, len(self.data.dictionary),
                       self.args.emsize,
                       self.args.nhid, self.args.lat_dim, self.args.nlayers,
                       dropout=self.args.dropout, tie_weights=self.args.tied)
        model.load_state_dict(torch.load(os.path.join(path, name + '.model')))
        model = model.cuda()
        return model

    def eva(self):
        # Load the best saved model.
        cur_loss, cur_kl, test_loss = self.evaluate(self.args, self.model,
                                                    self.data.test)
        Runner.log_eval(cur_loss, cur_kl, test_loss, True)

    def evaluate(self, args, model, dev_batches):
        """
        Standard evaluation function on dev or test set.
        :param args:
        :param model:
        :param dev_batches:
        :return:
        """

        # Turn on training mode which enables dropout.
        model.eval()
        model.FLAG_train = False

        acc_loss = 0
        acc_kl_loss = 0
        acc_total_loss = 0
        all_cnt = 0
        start_time = time.time()

        for idx, batch in enumerate(dev_batches):
            feed = self.data.get_feed(batch)
            target = GVar(batch)
            seq_len, batch_sz = batch.size()
            tup, kld, decoded = model(feed, target)

            flatten_decoded = decoded.view(-1, self.model.ntoken)
            flatten_target = target.view(-1)
            loss = self.criterion(flatten_decoded, flatten_target)  # batch_sz * seq, loss
            sum_kld = torch.sum(kld)
            total_loss = loss + sum_kld * self.args.kl_weight

            acc_total_loss += loss.data * seq_len * batch_sz + sum_kld.data
            acc_loss += loss.data * seq_len * batch_sz
            acc_kl_loss += sum_kld.data
            all_cnt += batch_sz * seq_len

        # word ppl
        cur_loss = acc_loss[0] / all_cnt  # word loss
        cur_kl = acc_kl_loss[0] / all_cnt
        # cur_real_loss = acc_real_loss / doc_cnt
        cur_real_loss = cur_loss + cur_kl
        elapsed = time.time() - start_time

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
        acc_total_loss = 0

        batch_cnt = 0
        all_cnt = 0
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

            tup, kld, decoded = model(feed, target)

            flatten_decoded = decoded.view(-1, self.model.ntoken)
            flatten_target = target.view(-1)
            loss = self.criterion(flatten_decoded, flatten_target)  # seq, loss
            detail_loss = self.detail_crit(flatten_decoded, flatten_target) \
                .view(seq_len, batch_sz)
            sum_kld = torch.sum(kld)
            vs.add_batch(target.data, tup, kld.data, detail_loss.data)

            acc_total_loss += loss.data * seq_len * batch_sz + sum_kld.data
            acc_loss += loss.data * seq_len * batch_sz
            acc_kl_loss += sum_kld.data

            batch_cnt += 1
            all_cnt += batch_sz * seq_len

        cur_loss = acc_loss[0] / all_cnt
        cur_kl = acc_kl_loss[0] / all_cnt
        # cur_real_loss = acc_real_loss / doc_cnt
        cur_real_loss = acc_total_loss[0] / all_cnt
        Runner.log_instant(None, self.args, glob_iter, epo, start_time, cur_loss
                           , cur_kl,
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

    def add_batch(self, target, tup, kld, loss):
        seq_len, batch_sz = loss.size()
        _seq_len, _batch_sz = target.size()
        # __batch = kld.size()[0]
        assert seq_len == _seq_len
        assert batch_sz == _batch_sz
        mu = tup['mu']
        # print(target.size())
        # print(batch_sz)
        for b in range(batch_sz):
            this_target = target[:, b]
            this_mu = mu[b]
            this_loss = loss[:, b]
            self.add_single(this_target, this_mu,
                            this_loss)

    def add_single(self, target, mu, loss):
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
        with open('v_european.txt', 'w') as f:
            f.write('\n'.join(self.logs))


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
            x = numpy.asarray(nums)
            bag.append(x)
        bags.append(bag)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def _mean_of_bag(bag):
        x = 0
        for b in range(len(bag)):
            x += bag[b]
        return x / len(bag)

    def _comp(_b):
        distance = 0
        cnt = 0
        for i in range(len(_b)):
            for j in range(i, len(_b)):
                bi = torch.FloatTensor(_b[i])
                bj = torch.FloatTensor(_b[j])
                x = torch.sum(bi * bj) / (torch.norm(bi) * torch.norm(bj))
                # print(bi.size())
                # print(dis)
                distance += x
                cnt += 1.
        return distance / cnt

    print(_comp(bags[0]))
    print(_comp(bags[1]))

    bag0 = torch.FloatTensor(_mean_of_bag(bags[0]))
    bag1 = torch.FloatTensor(_mean_of_bag(bags[1]))
    x = torch.sum(bag0 * bag1) / (torch.norm(bag1) * torch.norm(bag0))
    print(x)
    # print(ds.cosine(_mean_of_bag(bags[0]), _mean_of_bag(bags[1])))


if __name__ == '__main__':
    # player = PlayNVRNN('/home/jcxu/vae_txt/NVLL',
    #                    'Dataptb_Distvmf_Modelnvrnn_Emb500_Hid500_lat200_lr0.001_drop0.4'
    #                    , '/home/jcxu/vae_txt/data/ptb')
    # # player.eva()
    # player.play_eval(player.args, player.model, player.data.demo, 0, 0, 0)

    os.chdir('/home/jcxu/vae_txt/NVLL/framework')
    compute_cos(['ve.txt', 'vh.txt'])
