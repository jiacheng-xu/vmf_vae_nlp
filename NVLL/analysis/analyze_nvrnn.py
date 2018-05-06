# Load nvdm or nvrnn, check how Gauss or vMF distributes
"Dataptb_Distnor_Modelnvrnn_Emb400_Hid400_lat200_lr0.001_drop0.2"

import logging
import os
import shutil
import time

import numpy
import torch

from NVLL.analysis.analyzer_argparse import parse_arg
from NVLL.data.lm import DataLM
from NVLL.model.nvrnn import RNNVAE
from NVLL.util.util import GVar, swap_by_batch, replace_by_batch


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

        wt_str = "gt\tpred\trecon_nll\tkl\ttotal_nll\n{}\n{}\n{}\n{}\n{}\n".format(self.gt, self.pred, self.recon_nll, self.kl,
                                                   self.total_nll
                                                   )
        if self.dist_type == 'nor':
            wt_str += "{}\n{}\n{}".format(list_to_str(self.code),list_to_str(self.mean), list_to_str(self.logvar))
        elif self.dist_type == 'vmf':
            wt_str += "{}\n{}".format(list_to_str(self.code),list_to_str(self.mu))
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
        out = torch.multinomial(prob,1)
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
                if self.model.condition:
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
        cur_aux_loss = acc_aux_loss[0] / all_cnt
        cur_avg_cos = acc_avg_cos[0] / cnt
        cur_avg_norm = acc_avg_norm[0] / cnt
        cur_real_loss = cur_loss + cur_kl
        return cur_loss, cur_kl, cur_real_loss


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
    instance.logger.info("{}\t{}\t{}".format(cur_loss, cur_kl, cur_real_loss))
    with open(os.path.join(args.exp_path,args.board),'a' )as fd:
        fd.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            args.data_path, args.instance_name, args.mix_unk,
            args.swap, args.replace ,args.cd_bow,args.cd_bit,cur_loss,cur_kl, cur_real_loss,
            numpy.math.exp(cur_real_loss)))