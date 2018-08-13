# Load nvdm or nvrnn, check how Gauss or vMF distributes
"Dataptb_Distnor_Modelnvrnn_Emb400_Hid400_lat200_lr0.001_drop0.2"

import logging
import os
import random
import shutil
import time

import numpy
import torch

from NVLL.analysis.analyzer_argparse import parse_arg
from NVLL.data.lm import DataLM
from NVLL.model.nvrnn import RNNVAE
from NVLL.util.util import GVar, swap_by_batch, replace_by_batch, replace_by_batch_with_unk

cos = torch.nn.CosineSimilarity()


class Sample():
    def __init__(self, gt, pred, code, recon_nll, kl):
        self.gt = gt
        self.pred = pred
        self.code = code
        self.recon_nll = recon_nll
        self.kl = kl
        self.total_nll = recon_nll + kl
        # self.ppl = math.exp(self.total_nll)

    def set_nor_stat(self, mean, logvar):
        self.dist_type = "nor"
        self.mean = mean
        self.logvar = logvar

    def set_vmf_stat(self, mu):
        self.dist_type = "vmf"
        self.mu = mu

    def set_zero_stat(self):
        self.dist_type = "zero"

    def __repr__(self):
        def list_to_str(l):
            l = [str(i) for i in l]
            return "\t".join(l)

        wt_str = "gt\tpred\trecon_nll\tkl\ttotal_nll\n{}\n{}\n{}\n{}\n{}\n".format(self.gt, self.pred, self.recon_nll,
                                                                                   self.kl,
                                                                                   self.total_nll
                                                                                   )
        if self.dist_type == 'nor':
            wt_str += "{}\n{}\n{}".format(list_to_str(self.code), list_to_str(self.mean), list_to_str(self.logvar))
        elif self.dist_type == 'vmf':
            wt_str += "{}\n{}".format(list_to_str(self.code), list_to_str(self.mu))
        return wt_str


class ExpAnalyzer():
    def __init__(self, root_path="/home/cc/vae_txt",
                 exp_path="/home/cc/exp-nvrnn",
                 instance_name=None,
                 data_path="data/ptb",
                 eval_batch_size=5,
                 mix_unk=0,
                 swap=0, replace=0,
                 cd_bow=0, cd_bit=0):
        self.exp_path = exp_path
        self.instance_name = instance_name
        self.temp = 1
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(exp_path, instance_name + ".log"))
        ch = logging.StreamHandler()
        logger.addHandler(fh)
        logger.addHandler(ch)
        self.logger = logger
        self.logger.info("Loading File: {}".format(os.path.join(exp_path, instance_name)))
        self.args = self.load_args(exp_path, instance_name)

        self.logger.info(
            "Pre config: Swap:{}\tReplace:{}\tMixUnk:{}\tCdBit:{}\tCdBoW:{}\nLoaded Hyper-param WILL overwrite pre-config.".format(
                self.args.swap, self.args.replace, self.args.mix_unk, self.args.cd_bit, self.args.cd_bow))
        self.logger.info("Post config: Swap:{}\tReplace:{}\tMixUnk:{}\tCdBit:{}\tCdBoW:{}".format(
            swap, replace, mix_unk, cd_bit, cd_bow
        ))
        if swap != 0:
            self.args.swap = swap
        if replace != 0:
            self.args.replace = replace
        if mix_unk != 0:
            self.args.mix_unk = mix_unk
        if cd_bow > 1 and cd_bow != self.args.cd_bow:
            self.logger.warning("Unexpected chage: CD BoW")
            self.args.cd_bow = cd_bow
        if cd_bit > 1 and cd_bit != self.args.cd_bit:
            self.logger.warning("Unexpected chage: CD Bit")
            self.args.cd_bit = cd_bit

        self.data = self.load_data(os.path.join(root_path, data_path), eval_batch_size, cd_bit > 1)
        self.model = self.load_model(self.args, len(self.data.dictionary), exp_path, instance_name)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.crit_sample = torch.nn.CrossEntropyLoss(ignore_index=0, reduce=False)

    @staticmethod
    def load_args(path, name):
        with open(os.path.join(path, name + '.args'), 'rb') as f:
            args = torch.load(f)
        return args

    @staticmethod
    def load_data(data_path, eval_batch_siez, condition):
        data = DataLM(data_path, eval_batch_siez, eval_batch_siez, condition)
        return data

    @staticmethod
    def load_model(args, ntoken, path, name):
        model = RNNVAE(args, args.enc_type, ntoken, args.emsize,
                       args.nhid, args.lat_dim, args.nlayers,
                       dropout=args.dropout, tie_weights=args.tied,
                       input_z=args.input_z, mix_unk=args.mix_unk,
                       condition=(args.cd_bit or args.cd_bow),
                       input_cd_bow=args.cd_bow, input_cd_bit=args.cd_bit)
        model.load_state_dict(torch.load(os.path.join(path, name + '.model')))
        from NVLL.util.gpu_flag import GPU_FLAG
        if torch.cuda.is_available() and GPU_FLAG:
            model = model.cuda()
        model = model.eval()
        return model

    def analyze_batch(self, target, kld, tup, vecs, decoded):
        _tmp_bag = []
        seq_len, batch_sz = target.size()
        for b in range(batch_sz):
            gt = target[:, b]
            deco = decoded[:, b, :]
            if self.model.dist_type == 'zero':
                sample = self.analyze_zero(gt, deco)
            else:
                kl = kld[b]
                lat_code = vecs[:, b, :]
                if self.model.dist_type == 'vmf':
                    mu = tup['mu'][b]
                    sample = self.analyze_vmf(gt, kl, mu, lat_code, deco)
                elif self.model.dist_type == 'nor':
                    mean = tup['mean'][b]
                    logvar = tup['logvar'][b]
                    sample = self.analyze_nor(gt, kl, mean, logvar, lat_code, deco)
                else:
                    raise NotImplementedError
            _tmp_bag.append(sample)
        return _tmp_bag

    def analyze_batch_order(self, original_vecs, manipu_vecs):
        """
        Given original codes and manipilated codes, comp their cos similarity
        :param original_vecs: sample_num, batch_size, lat_code
        :param manipu_vecs: sample_num, batch_size, lat_code
        :return:
        """
        original_vecs = torch.mean(original_vecs, dim=0).unsqueeze(2)
        manipu_vecs = torch.mean(manipu_vecs, dim=0).unsqueeze(2)

        x = cos(original_vecs, manipu_vecs)
        # print(x)
        return torch.mean(x.squeeze())

    def analyze_batch_word_importance(self, original_vecs, manipu_vecs, masked_words):
        pass

    def analyze_batch_order_and_importance(self, original_vecs, manipulated_vecs):
        _tmp_bag = []

        batch_sz = original_vecs.size()[1]
        for b in range(batch_sz):
            if self.model.dist_type == 'zero':
                raise NotImplementedError
            else:
                lat_code = vecs[:, b, :]
                lat_code = torch.mean(lat_code, dim=0)
                if self.model.dist_type == 'vmf':
                    mu = tup['mu']
                    sample = self.analyze_vmf(gt, kl, mu, lat_code, deco)
                elif self.model.dist_type == 'nor':
                    mean = tup['mean'][b]
                    logvar = tup['logvar'][b]
                    sample = self.analyze_nor(gt, kl, mean, logvar, lat_code, deco)
                else:
                    raise NotImplementedError
            _tmp_bag.append(sample)
        return _tmp_bag

    def analyze_zero(self, gt, decoded):
        pred_id = self.decode_to_ids(decoded)
        gt_id = gt.data.tolist()
        pred_words = self.ids_to_words(pred_id)
        gt_words = self.ids_to_words(gt_id)
        recon_nll = self.criterion(decoded, gt).data[0]
        s = Sample(gt=gt_words,
                   pred=pred_words,
                   code=None,
                   recon_nll=recon_nll, kl=0)
        s.set_zero_stat()
        return s

    def analyze_vmf(self, gt, kld, mu, lat_code, decoded):
        pred_id = self.decode_to_ids(decoded)
        gt_id = gt.data.tolist()
        pred_words = self.ids_to_words(pred_id)
        gt_words = self.ids_to_words(gt_id)
        kl_val = kld.data[0]
        mu = mu.data.tolist()
        lat_code = torch.mean(lat_code, dim=0)
        lat_code = lat_code.data.tolist()
        recon_nll = self.criterion(decoded, gt).data[0]
        s = Sample(gt=gt_words,
                   pred=pred_words,
                   code=lat_code,
                   recon_nll=recon_nll, kl=kl_val)
        s.set_vmf_stat(mu)
        return s

    def analyze_nor(self, gt, kld, mean, logvar, lat_code, decoded):
        pred_id = self.decode_to_ids(decoded)
        gt_id = gt.data.tolist()
        pred_words = self.ids_to_words(pred_id)
        gt_words = self.ids_to_words(gt_id)
        kl_val = kld.data[0]
        mean = mean.data.tolist()
        logvar = logvar.data.tolist()
        lat_code = torch.mean(lat_code, dim=0)
        lat_code = lat_code.data.tolist()
        recon_nll = self.criterion(decoded, gt).data[0]
        s = Sample(gt=gt_words,
                   pred=pred_words,
                   code=lat_code,
                   recon_nll=recon_nll, kl=kl_val)
        s.set_nor_stat(mean, logvar)
        return s

    def decode_to_ids(self, prob):
        seq_len, vocab_szie = prob.size()
        assert vocab_szie == len(self.data.dictionary)
        prob = torch.exp(prob).div(self.temp)
        out = torch.multinomial(prob, 1)
        # _, argmax = torch.max(prob, dim=1, keepdim=False)
        ids = out.data.tolist()
        ids = [x[0] for x in ids]
        return ids

    def ids_to_words(self, ids):
        words = []
        for i in ids:
            words.append(self.data.dictionary.query(i))
        return " ".join(words)

    def write_samples(self, bag):
        if os.path.exists(os.path.join(self.exp_path, self.instance_name + 'logs')):
            shutil.rmtree(os.path.join(self.exp_path, self.instance_name + 'logs'))
        os.mkdir(os.path.join(self.exp_path, self.instance_name + 'logs'))
        self.logger.info("Logs path: {}".format(os.path.join(self.exp_path, self.instance_name + 'logs')))
        os.chdir(os.path.join(self.exp_path, self.instance_name + 'logs'))
        for idx, b in enumerate(bag):
            with open("log-" + str(idx) + '.txt', 'w') as fd:
                fd.write(repr(b))

    def analysis_evaluation(self):
        self.logger.info("Start Analyzing ...")
        start_time = time.time()
        test_batches = self.data.test
        self.logger.info("Total {} batches to analyze".format(len(test_batches)))
        acc_loss = 0
        acc_kl_loss = 0
        acc_aux_loss = 0
        acc_avg_cos = 0
        acc_avg_norm = 0

        batch_cnt = 0
        all_cnt = 0
        cnt = 0
        sample_bag = []
        try:
            for idx, batch in enumerate(test_batches):
                if idx % 10 == 0:
                    print("Idx: {}".format(idx))
                seq_len, batch_sz = batch.size()
                if self.data.condition:
                    seq_len -= 1
                    bit = batch[0, :]
                    batch = batch[1:, :]
                    bit = GVar(bit)
                else:
                    bit = None
                feed = self.data.get_feed(batch)

                if self.args.swap > 0.00001:
                    feed = swap_by_batch(feed, self.args.swap)
                if self.args.replace > 0.00001:
                    feed = replace_by_batch(feed, self.args.replace, self.model.ntoken)

                target = GVar(batch)

                recon_loss, kld, aux_loss, tup, vecs, decoded = self.model(feed, target, bit)
                # target: seq_len, batchsz
                # decoded: seq_len, batchsz, dict_sz
                # tup: 'mean' 'logvar' for Gaussian
                #         'mu' for vMF
                # vecs
                bag = self.analyze_batch(target, kld, tup, vecs, decoded)
                sample_bag += bag
                acc_loss += recon_loss.data * seq_len * batch_sz
                acc_kl_loss += torch.sum(kld).data
                acc_aux_loss += torch.sum(aux_loss).data
                acc_avg_cos += tup['avg_cos'].data
                acc_avg_norm += tup['avg_norm'].data
                cnt += 1
                batch_cnt += batch_sz
                all_cnt += batch_sz * seq_len
        except KeyboardInterrupt:
            print("early stop")
        self.write_samples(sample_bag)
        cur_loss = acc_loss[0] / all_cnt
        cur_kl = acc_kl_loss[0] / all_cnt
        cur_real_loss = cur_loss + cur_kl
        return cur_loss, cur_kl, cur_real_loss

    def analysis_eval_word_importance(self, feed, batch, bit):
        """
        Given a sentence, replace a certain word by UNK and see how lat code change from the origin one.
        :param feed:
        :param batch:
        :param bit:
        :return:
        """
        seq_len, batch_sz = batch.size()
        target = GVar(batch)
        origin_feed = feed.clone()
        original_recon_loss, kld, _, original_tup, original_vecs, _ = self.model(origin_feed, target, bit)
        # original_vecs = torch.mean(original_vecs, dim=0).unsqueeze(2)
        original_mu = original_tup['mu']
        # table_of_code = torch.FloatTensor(seq_len, batch_sz )
        table_of_mu = torch.FloatTensor(seq_len, batch_sz)
        for t in range(seq_len):
            cur_feed = feed.clone()
            cur_feed[t, :] = 2
            cur_recon, _, _, cur_tup, cur_vec, _ = self.model(cur_feed, target, bit)

            cur_mu = cur_tup['mu']
            # cur_vec = torch.mean(cur_vec, dim=0).unsqueeze(2)
            # x = cos(original_vecs, cur_vec)
            # x= x.squeeze()
            y = cos(original_mu, cur_mu)
            y = y.squeeze()

            # table_of_code[t,:] = x.data
            table_of_mu[t, :] = y.data
        bag = []
        for b in range(batch_sz):
            weight = table_of_mu[:, b]
            word_ids = feed[:, b]
            words = self.ids_to_words(word_ids.data.tolist())
            seq_of_words = words.split(" ")
            s = ""
            for t in range(seq_len):
                if weight[t] < 0.98:
                    s += "*" + seq_of_words[t] + "* "
                else:
                    s += seq_of_words[t] + " "
            bag.append(s)
        return bag

    def analysis_eval_order(self, feed, batch, bit):
        assert 0.33 > self.args.swap > 0.0001
        origin_feed = feed.clone()

        feed_1x = swap_by_batch(feed.clone(), self.args.swap)
        feed_2x = swap_by_batch(feed.clone(), self.args.swap * 2)
        feed_3x = swap_by_batch(feed.clone(), self.args.swap * 3)
        feed_4x = swap_by_batch(feed.clone(), self.args.swap * 4)
        feed_5x = swap_by_batch(feed.clone(), self.args.swap * 5)
        feed_6x = swap_by_batch(feed.clone(), self.args.swap * 6)
        target = GVar(batch)

        # recon_loss, kld, aux_loss, tup, vecs, decoded = self.model(feed, target, bit)
        original_recon_loss, kld, _, original_tup, original_vecs, _ = self.model(origin_feed, target, bit)
        if 'Distnor' in self.instance_name:
            key_name = "mean"
        elif 'vmf' in self.instance_name:
            key_name = "mu"
        else:
            raise NotImplementedError

        original_mu = original_tup[key_name]
        recon_loss_1x, _, _, tup_1x, vecs_1x, _ = self.model(feed_1x, target, bit)
        recon_loss_2x, _, _, tup_2x, vecs_2x, _ = self.model(feed_2x, target, bit)
        recon_loss_3x, _, _, tup_3x, vecs_3x, _ = self.model(feed_3x, target, bit)
        recon_loss_4x, _, _, tup_4x, vecs_4x, _ = self.model(feed_4x, target, bit)
        recon_loss_5x, _, _, tup_5x, vecs_5x, _ = self.model(feed_5x, target, bit)
        recon_loss_6x, _, _, tup_6x, vecs_6x, _ = self.model(feed_6x, target, bit)

        # target: seq_len, batchsz
        # decoded: seq_len, batchsz, dict_sz
        # tup: 'mean' 'logvar' for Gaussian
        #         'mu' for vMF
        # vecs
        # cos_1x = self.analyze_batch_order(original_vecs, vecs_1x).data
        # cos_2x = self.analyze_batch_order(original_vecs, vecs_2x).data
        # cos_3x = self.analyze_batch_order(original_vecs, vecs_3x).data
        cos_1x = torch.mean(cos(original_mu, tup_1x[key_name])).data
        cos_2x = torch.mean(cos(original_mu, tup_2x[key_name])).data
        cos_3x = torch.mean(cos(original_mu, tup_3x[key_name])).data
        cos_4x = torch.mean(cos(original_mu, tup_4x[key_name])).data
        cos_5x = torch.mean(cos(original_mu, tup_5x[key_name])).data
        cos_6x = torch.mean(cos(original_mu, tup_6x[key_name])).data
        # print(cos_1x, cos_2x, cos_3x)
        return [
            [original_recon_loss.data, recon_loss_1x.data, recon_loss_2x.data, recon_loss_3x.data, recon_loss_4x.data,
             recon_loss_5x.data, recon_loss_6x.data]
            , [cos_1x, cos_2x, cos_3x, cos_4x, cos_5x, cos_6x]]

    def unpack_bag_order(self, sample_bag):
        import numpy as np
        l = len(sample_bag)
        print("Total {} batches".format(l))
        acc_loss = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
        acc_cos = np.asarray([0., 0., 0., 0, 0, 0])
        acc_cnt = 0
        # print(sample_bag)
        for b in sample_bag:
            acc_cnt += 1
            losses = b[0]
            # print(losses)
            for idx, x in enumerate(losses):
                acc_loss[idx] += x

            _cos = b[1]
            # print(b[1])
            acc_cos += np.asarray(_cos)
            # for idx, x in enumerate(_cos):
            #     acc_cos[idx] += np.asarray(x[0])

        acc_loss = [x / acc_cnt for x in acc_loss]
        acc_cos = [x / acc_cnt for x in acc_cos]
        instance.logger.info("-" * 50)
        instance.logger.info(
            "Origin Loss|1x|2x|3x|4x:\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(acc_loss[0], acc_loss[1], acc_loss[2],
                                                                            acc_loss[3], acc_loss[4],
                                                                            acc_loss[5], acc_loss[6]))

        instance.logger.info(
            "Cos   1x|2x|3x|4x|5x|6x:\t{}\t{}\t{}\t{}\t{}\t{}\n".format(acc_cos[0], acc_cos[1], acc_cos[2], acc_cos[3],
                                                                        acc_cos[4], acc_cos[5]))
        return acc_cos, acc_loss

    def unpack_bag_word_importance(self, sample_bag):
        for b in sample_bag:
            for x in b:
                print(x)
                print("-" * 80)

    def analysis_evaluation_order_and_importance(self):
        """
        Measure the change of cos sim given different encoding sequence
        :return:
        """
        self.logger.info("Start Analyzing ... Picking up 100 batches to analyze")
        start_time = time.time()
        test_batches = self.data.test
        random.shuffle(test_batches)
        test_batches = test_batches[:100]
        self.logger.info("Total {} batches to analyze".format(len(test_batches)))
        acc_loss = 0
        acc_kl_loss = 0

        batch_cnt = 0
        all_cnt = 0
        cnt = 0
        sample_bag = []
        try:
            for idx, batch in enumerate(test_batches):
                if idx % 10 == 0:
                    print("Now Idx: {}".format(idx))
                seq_len, batch_sz = batch.size()
                if self.data.condition:
                    seq_len -= 1
                    bit = batch[0, :]
                    batch = batch[1:, :]
                    bit = GVar(bit)
                else:
                    bit = None
                feed = self.data.get_feed(batch)

                if self.args.swap > 0.0001:
                    bag = self.analysis_eval_order(feed, batch, bit)
                elif self.args.replace > 0.0001:
                    bag = self.analysis_eval_word_importance(feed, batch, bit)
                else:
                    print("Maybe Wrong mode?")
                    raise NotImplementedError

                sample_bag.append(bag)
        except KeyboardInterrupt:
            print("early stop")
        if self.args.swap > 0.0001:
            return self.unpack_bag_order(sample_bag)
        elif self.args.replace > 0.0001:
            return self.unpack_bag_word_importance(sample_bag)
        else:
            raise NotImplementedError


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
    args = parse_arg()
    instance = ExpAnalyzer(root_path=args.root_path,
                           exp_path=args.exp_path,
                           instance_name=
                           # "Datayelp_Distvmf_Modelnvrnn_EnclstmBiFalse_Emb100_Hid400_lat100_lr10.0_drop0.5_kappa40.0_auxw0.0001_normfFalse_nlay1_mixunk0.0_inpzTrue_cdbit50_cdbow0"
                           args.instance_name
                           ,
                           data_path=args.data_path,
                           eval_batch_size=args.eval_batch_size,
                           mix_unk=args.mix_unk,
                           swap=args.swap, replace=args.replace,
                           cd_bow=args.cd_bow, cd_bit=args.cd_bit)
    cur_loss, cur_kl, cur_real_loss = instance.analysis_evaluation()

    # instance.logger.info("{}\t{}\t{}".format(cur_loss, cur_kl, cur_real_loss))
    # print(cur_loss, cur_kl, cur_real_loss, numpy.math.exp(cur_real_loss))
    # with open(os.path.join(args.exp_path,args.board),'a' )as fd:
    #     fd.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
    #         args.data_path, args.instance_name, args.mix_unk,
    #         args.swap, args.replace ,args.cd_bow,args.cd_bit,cur_loss,cur_kl, cur_real_loss,
    #         numpy.math.exp(cur_real_loss)))

    acc_cos, acc_loss = instance.analysis_evaluation_order_and_importance()
    with open(os.path.join(args.exp_path, args.board), 'a')as fd:
        fd.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            args.data_path, args.instance_name, args.mix_unk,
            args.swap, args.replace, args.cd_bow, args.cd_bit,
            acc_loss[0], acc_loss[1], acc_loss[2], acc_loss[3], acc_cos[0], acc_cos[1], acc_cos[2]))

    # "--data_path data/yelp --swap 0 --replace 0 --cd_bit 50 --root_path /home/cc/vae_txt --exp_path /home/cc/save-nvrnn --instance_name   Datayelp_Distvmf_Modelnvrnn_EnclstmBiFalse_Emb100_Hid400_lat100_lr10.0_drop0.5_kappa200.0_auxw0.0001_normfFalse_nlay1_mixunk1.0_inpzTrue_cdbit50_cdbow0_4.9021353610814655   --cd_bow 	0   --mix_unk   1"
