# Load nvdm or nvrnn, check how Gauss or vMF distributes
"Dataptb_Distnor_Modelnvrnn_Emb400_Hid400_lat200_lr0.001_drop0.2"

import logging
import os
import random
import shutil
import time

import numpy as np
import torch

from NVLL.analysis.analyzer_argparse import parse_arg
from NVLL.data.lm import DataLM
from NVLL.model.nvrnn import RNNVAE
from NVLL.util.util import GVar, swap_by_batch, replace_by_batch

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
            return " ".join(l)

        wt_str = "GT_PD_RecNLL_KL_NLL_code_mu_or_mean_logvar\t{}\t{}\t{}\t{}\t{}\t".format(self.gt, self.pred,
                                                                                           self.recon_nll,
                                                                                           self.kl,
                                                                                           self.total_nll
                                                                                           )
        if self.dist_type == 'nor':
            wt_str += "{}\t{}\t{}".format(list_to_str(self.code), list_to_str(self.mean), list_to_str(self.logvar))
        elif self.dist_type == 'vmf':
            wt_str += "{}\t{}".format(list_to_str(self.code), list_to_str(self.mu))
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
        from NVLL.util.gpu_flag import device
        # if torch.cuda.is_available() and GPU_FLAG:
        #     model = model.cuda()
        model.to(device
                 )
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
            if i == 0:
                continue
            words.append(self.data.dictionary.query(i))
        return " ".join(words)

    def write_samples(self, bag, name):
        # if os.path.exists(os.path.join(self.exp_path, self.instance_name + 'logs')):
        #     shutil.rmtree(os.path.join(self.exp_path, self.instance_name + 'logs'))
        # os.mkdir(os.path.join(self.exp_path, self.instance_name + 'logs'))
        # self.logger.info("Logs path: {}".format(os.path.join(self.exp_path, self.instance_name + 'logs')))
        # os.chdir(os.path.join(self.exp_path, self.instance_name + 'logs'))
        log_file_name = os.path.join(self.exp_path, self.instance_name + 'logs_' + name + '.txt')
        writes = []
        for idx, b in enumerate(bag):
            s = repr(b)
            writes.append(s)

        with open(log_file_name, 'w') as fd:
            fd.write("\n".join(writes))
            print("Writing finish! {}".format(log_file_name))

    def analysis_evaluation(self, test_batch, name):
        self.logger.info("Start Analyzing ...")

        # train_batches = self.data.train
        # dev_batches = self.data.dev
        # test_batches = self.data.test
        self.logger.info("Total {} batches to analyze".format(len(test_batch)))
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
            for idx, batch in enumerate(test_batch):
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
        self.write_samples(sample_bag, name)
        cur_loss = acc_loss[0] / all_cnt
        cur_kl = acc_kl_loss[0] / all_cnt
        cur_real_loss = cur_loss + cur_kl
        return cur_loss, cur_kl, cur_real_loss


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
    if args.split == 0:
        cur_loss, cur_kl, cur_real_loss = instance.analysis_evaluation(instance.data.test, 'test')
    elif args.split == 1:
        cur_loss, cur_kl, cur_real_loss = instance.analysis_evaluation(instance.data.dev, 'dev')
    elif args.split == 2:
        cur_loss, cur_kl, cur_real_loss = instance.analysis_evaluation(instance.data.train, 'train')
    else:
        raise NotImplementedError

    # instance.logger.info("{}\t{}\t{}".format(cur_loss, cur_kl, cur_real_loss))
    print(cur_loss, cur_kl, cur_real_loss, np.math.exp(cur_real_loss))
    # with open(os.path.join(args.exp_path,args.board),'a' )as fd:
    #     fd.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
    #         args.data_path, args.instance_name, args.mix_unk,
    #         args.swap, args.replace ,args.cd_bow,args.cd_bit,cur_loss,cur_kl, cur_real_loss,
    #         numpy.math.exp(cur_real_loss)))

    # "--data_path data/yelp --swap 0 --replace 0 --cd_bit 50 --root_path /home/cc/vae_txt --exp_path /home/cc/save-nvrnn --instance_name   Datayelp_Distvmf_Modelnvrnn_EnclstmBiFalse_Emb100_Hid400_lat100_lr10.0_drop0.5_kappa200.0_auxw0.0001_normfFalse_nlay1_mixunk1.0_inpzTrue_cdbit50_cdbow0_4.9021353610814655   --cd_bow 	0   --mix_unk   1"
