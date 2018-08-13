import argparse
import os
from operator import itemgetter

import torch

from NVLL.data.lm import DataLM
from NVLL.model.nvrnn import RNNVAE
from NVLL.util.util import GVar


def load_args(path, name):
    with open(os.path.join(path, name + '.args'), 'rb') as f:
        args = torch.load(f)
    return args


def load_data(data_path, eval_batch_siez, condition):
    data = DataLM(data_path, eval_batch_siez, eval_batch_siez, condition)
    return data


def load_model(args, ntoken, path, name):
    model = RNNVAE(args, args.enc_type, ntoken, args.emsize,
                   args.nhid, args.lat_dim, args.nlayers,
                   dropout=args.dropout, tie_weights=args.tied,
                   input_z=args.input_z, mix_unk=args.mix_unk,
                   condition=(args.cd_bit or args.cd_bow),
                   input_cd_bow=args.cd_bow, input_cd_bit=args.cd_bit)
    print("Loading {}".format(name))
    model.load_state_dict(torch.load(os.path.join(path, name + '.model')))
    from NVLL.util.gpu_flag import GPU_FLAG
    if torch.cuda.is_available() and GPU_FLAG:
        model = model.cuda()
    model = model.eval()
    return model


def parse_arg():
    parser = argparse.ArgumentParser(description='Transfer experiment')
    parser.add_argument('--data_path', type=str, default='data/ptb', help='location of the data corpus')
    parser.add_argument('--root_path', type=str, default='/home/jcxu/vae_txt')
    parser.add_argument('--model_vmf', type=str,
                        default="Dataptb_Distvmf_Modelnvrnn_EnclstmBiFalse_Emb100_Hid400_lat50_lr10.0_drop0.5_kappa120.0_auxw0.0001_normfFalse_nlay1_mixunk1.0_inpzTrue_cdbit0_cdbow0_ann0_5.891579498848754")
    parser.add_argument('--model_nor', type=str,
                        default=
                        "Dataptb_Distnor_Modelnvrnn_EnclstmBiFalse_Emb100_Hid400_lat50_lr10.0_drop0.5_kappa0.1_auxw0.0001_normfFalse_nlay1_mixunk1.0_inpzTrue_cdbit0_cdbow0_ann2_5.933433308374706")
    parser.add_argument('--exp_path', type=str, default='/backup2/jcxu/exp-nvrnn')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='evaluation batch size')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')

    args = parser.parse_args()
    return args


class Transfer():
    @staticmethod
    def write_word_embedding(exp_path, file_name, word_list, embedding_mat):
        embedding_mat = embedding_mat.data
        path = os.path.join(exp_path, file_name)
        print("To save {}".format(os.path.join(exp_path, file_name)))
        bag = []
        for idx, w in enumerate(word_list):
            name = w[0]
            emb = embedding_mat[idx]
            l = [name] + emb.tolist()
            l = [str(x) for x in l]
            l = " ".join(l)
            bag.append(l)

        s = "\n".join(bag)
        with open(path, 'w') as fd:
            fd.write(s)

    def __init__(self, args):
        self.data = DataLM(os.path.join(args.root_path, args.data_path),
                           args.batch_size,
                           args.eval_batch_size,
                           condition=True)
        word_list = sorted(self.data.dictionary.word2idx.items(), key=itemgetter(1))

        vmf_args = load_args(args.exp_path, args.model_vmf)
        vmf_model = load_model(vmf_args, len(self.data.dictionary), args.exp_path, args.model_vmf)
        vmf_emb = vmf_model.emb.weight
        self.write_word_embedding(args.exp_path, args.model_vmf + '_emb', word_list, vmf_emb)
        nor_args = load_args(args.exp_path, args.model_nor)
        nor_model = load_model(nor_args, len(self.data.dictionary), args.exp_path, args.model_nor)
        nor_emb = nor_model.emb.weight
        self.write_word_embedding(args.exp_path, args.model_nor + '_emb', word_list, nor_emb)


def synthesis_bow_rep(args):
    data = DataLM(os.path.join(args.root_path, args.data_path),
                  args.batch_size,
                  args.eval_batch_size,
                  condition=True)


import random


class Code2Code(torch.nn.Module):
    def __init__(self, inp_dim, tgt_dim):
        super().__init__()
        self.linear = torch.nn.Linear(inp_dim, tgt_dim)
        self.linear2 = torch.nn.Linear(tgt_dim, tgt_dim)

        self.loss_func = torch.nn.CosineEmbeddingLoss()

    def forward(self, inp, tgt):
        pred = self.linear(inp)
        pred = torch.nn.functional.tanh(pred)
        pred = self.linear2(pred)
        # print(pred.size())
        loss = 1 - torch.nn.functional.cosine_similarity(pred, tgt)
        loss = torch.mean(loss)
        return loss


class CodeLearner():
    def __init__(self, args, condition, c2b, nor):
        self.data = DataLM(os.path.join(args.root_path, args.data_path),
                           args.batch_size,
                           args.eval_batch_size,
                           condition=condition)
        self.c2b = c2b
        if nor:
            args.model_run = args.model_nor
        else:
            args.model_run = args.model_vmf
        self.args = load_args(args.exp_path, args.model_run)
        self.model = load_model(self.args, len(self.data.dictionary),
                                args.exp_path, args.model_run)
        self.learner = Code2Code(self.model.lat_dim, self.model.ninp)
        self.learner.cuda()
        self.optim = torch.optim.Adam(self.learner.parameters(), lr=0.001)

    def run_train(self):
        valid_acc = []
        for e in range(10):
            print("EPO: {}".format(e))
            self.train_epo(self.data.train)
            acc = self.evaluate(self.data.test)
            valid_acc.append(acc)
        return min(valid_acc)

    def train_epo(self, train_batches):
        self.learner.train()
        print("Epo start")
        acc_loss = 0
        cnt = 0

        random.shuffle(train_batches)
        for idx, batch in enumerate(train_batches):
            self.optim.zero_grad()
            seq_len, batch_sz = batch.size()
            if self.data.condition:
                seq_len -= 1

                if self.model.input_cd_bit > 1:
                    bit = batch[0, :]
                    bit = GVar(bit)
                else:
                    bit = None
                batch = batch[1:, :]
            else:
                bit = None
            feed = self.data.get_feed(batch)

            seq_len, batch_sz = feed.size()
            emb = self.model.drop(self.model.emb(feed))

            if self.model.input_cd_bit > 1:
                bit = self.model.enc_bit(bit)
            else:
                bit = None

            h = self.model.forward_enc(emb, bit)
            tup, kld, vecs = self.model.forward_build_lat(h)  # batchsz, lat dim
            if self.model.dist_type == 'vmf':
                code = tup['mu']
            elif self.model.dist_type == 'nor':
                code = tup['mean']
            else:
                raise NotImplementedError
            emb = torch.mean(emb, dim=0)
            if self.c2b:
                loss = self.learner(code, emb)
            else:
                loss = self.learner(code, emb)
            loss.backward()
            self.optim.step()
            acc_loss += loss.data[0]
            cnt += 1
            if idx % 400 == 0 and (idx > 0):
                print("Training {}".format(acc_loss / cnt))
                acc_loss = 0
                cnt = 0

    def evaluate(self, dev_batches):
        self.learner.eval()
        print("Test start")
        acc_loss = 0
        cnt = 0
        random.shuffle(dev_batches)
        for idx, batch in enumerate(dev_batches):
            self.optim.zero_grad()
            seq_len, batch_sz = batch.size()
            if self.data.condition:
                seq_len -= 1

                if self.model.input_cd_bit > 1:
                    bit = batch[0, :]
                    bit = GVar(bit)
                else:
                    bit = None
                batch = batch[1:, :]
            else:
                bit = None
            feed = self.data.get_feed(batch)

            seq_len, batch_sz = feed.size()
            emb = self.model.drop(self.model.emb(feed))

            if self.model.input_cd_bit > 1:
                bit = self.model.enc_bit(bit)
            else:
                bit = None

            h = self.model.forward_enc(emb, bit)
            tup, kld, vecs = self.model.forward_build_lat(h)  # batchsz, lat dim
            if self.model.dist_type == 'vmf':
                code = tup['mu']
            elif self.model.dist_type == 'nor':
                code = tup['mean']
            else:
                raise NotImplementedError
            emb = torch.mean(emb, dim=0)
            if self.c2b:
                loss = self.learner(code, emb)
            else:
                loss = self.learner(code, emb)
            acc_loss += loss.data[0]
            cnt += 1
            if idx % 400 == 0:
                acc_loss = 0
                cnt = 0
        # print("===============test===============")
        # print(acc_loss / cnt)
        print(acc_loss / cnt)
        return float(acc_loss / cnt)


if __name__ == '__main__':
    print("Transfer btw Learnt Code and learnt BoW. "
          "Assume data is Yelp and model is vMF or nor.")
    args = parse_arg()
    # t = Transfer(args)
    # Synthesis data
    bags = []
    for c2b in [True, False]:
        for nor in [True]:
            learn = CodeLearner(args, condition=False, c2b=c2b, nor=nor)
            result = learn.run_train()
            bags.append("c2b\t{}\tnor\t{}\tresult:{}\n".format(c2b, nor, result))
            print("c2b\t{}\tnor\t{}\tresult:{}".format(c2b, nor, result))
    print(args)
    print("=" * 100)
    for b in bags:
        print(b)
