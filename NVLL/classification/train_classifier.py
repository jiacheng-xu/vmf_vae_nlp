# File for transfer experiment.
# Learn an MLP to map from code to bow or bow to code.
# Both code and bow are fixed and only loaded here as data. Param of MLP is the stuff we learn here.

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
    from NVLL.util.gpu_flag import device
    # if torch.cuda.is_available() and GPU_FLAG:
    #     model = model.cuda()
    model.to(device)
    model = model.eval()
    return model


def parse_arg():
    parser = argparse.ArgumentParser(description='Transfer experiment')
    parser.add_argument('--data_path', type=str, default='data/trec', help='location of the data corpus')
    parser.add_argument('--root_path', type=str, default='/home/jcxu/vae_txt')
    parser.add_argument('--model_vmf', type=str,
                        default="/backup2/jcxu/exp-nvrnn/Datatrec_Distvmf_Modelnvrnn_EnclstmBiFalse_Emb100_Hid400_lat100_lr10.0_drop0.5_kappa50.0_auxw0.0001_normfFalse_nlay1_mixunk1.0_inpzTrue_cdbit0_cdbow0_ann0_5.093163068262161")
    parser.add_argument('--model_nor', type=str,
                        default=
                        "/backup2/jcxu/exp-nvrnn/Datayelp_sent_Distnor_Modelnvrnn_EnclstmBiFalse_Emb100_Hid400_lat100_lr10.0_drop0.5_kappa0.1_auxw0.0001_normfFalse_nlay1_mixunk1.0_inpzTrue_cdbit0_cdbow0_ann0_5.75607256214362")
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


import random


class Code2Bit(torch.nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.linear = torch.nn.Linear(inp_dim, inp_dim)
        self.linear2 = torch.nn.Linear(inp_dim, 50)

        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, inp, tgt):
        pred = self.linear(inp)
        pred = torch.nn.functional.sigmoid(pred)
        pred = self.linear2(pred)
        # print(pred.size())
        loss = self.loss_func(pred, tgt)
        return loss, pred


import numpy as np


class SentClassifier():
    def __init__(self, args, condition, nor):

        if nor:
            args.model_run = args.model_nor
        else:
            args.model_run = args.model_vmf
        self.instance = args.model_run
        self.args = load_args(args.exp_path, args.model_run)

        self.train = self.load_log("train")
        # self.dev = self.load_log("dev")
        self.test = self.load_log("test")
        lat_dim = self.test.shape[1] - 1

        self.learner = Code2Bit(lat_dim)
        self.learner.cuda()
        self.optim = torch.optim.SGD(self.learner.parameters(), lr=0.0001)

    def load_log(self, name="train"):
        # Load related files. return [ vec , label] pairs
        with open(self.instance + "logs_" + name + ".lab.txt", 'r') as fd:
            lines = fd.read().splitlines()
        bag = []
        random.shuffle(lines)
        for l in lines:
            units = l.split("\t")
            label = int(units[0])
            vec = units[8]
            nums = vec.split(" ")
            nums = [label] + [float(x) for x in nums]
            bag.append(nums)
        arr = np.asarray(bag)
        return arr

    def batchify(self, tensor, batch_sz=20):
        np.random.shuffle(tensor)
        bag = []
        nsamples, l = tensor.shape
        nbatches = nsamples // batch_sz
        for n in range(nbatches):
            slice = tensor[n * batch_sz: (n + 1) * batch_sz]
            bit = slice[:, 0]
            bit = torch.LongTensor(bit)
            vec = torch.FloatTensor(slice[:, 1:])
            bag.append([bit, vec])
        return bag

    def run_train(self):
        valid_acc = []
        for e in range(50):
            print("EPO: {}".format(e))

            train = self.batchify(self.train)

            self.train_epo(train)

            test = self.batchify(self.test)
            acc = self.evaluate(test)
            valid_acc.append(acc)
        return min(valid_acc)

    def train_epo(self, train_batches):
        self.learner.train()
        print("Epo start")
        acc_loss = 0
        acc_accuracy = 0
        all_cnt = 0
        cnt = 0

        random.shuffle(train_batches)

        for idx, batch in enumerate(train_batches):
            self.optim.zero_grad()
            bit, vec = batch
            bit = GVar(bit)
            vec = GVar(vec)
            # print(bit)
            loss, pred = self.learner(vec, bit)
            _, argmax = torch.max(pred, dim=1)
            loss.backward()
            self.optim.step()

            argmax = argmax.data
            bit = bit.data
            for jdx, num in enumerate(argmax):
                gt = bit[jdx]
                all_cnt += 1
                if gt == num:
                    acc_accuracy += 1

            acc_loss += loss.data[0]
            cnt += 1
            if idx % 400 == 0:
                print("Loss {}  \tAccuracy {}".format(acc_loss / cnt, acc_accuracy / all_cnt))
                acc_loss = 0
                cnt = 0

    def evaluate(self, dev_batches):
        self.learner.eval()
        print("Test start")
        acc_loss = 0
        acc_accuracy = 0
        all_cnt = 0
        cnt = 0
        random.shuffle(dev_batches)
        for idx, batch in enumerate(dev_batches):
            self.optim.zero_grad()
            bit, vec = batch
            bit = GVar(bit)
            vec = GVar(vec)
            # print(bit)
            loss, pred = self.learner(vec, bit)
            _, argmax = torch.max(pred, dim=1)
            loss.backward()
            self.optim.step()

            argmax = argmax.data
            bit = bit.data
            for idx, num in enumerate(argmax):
                gt = bit[idx]
                all_cnt += 1
                if gt == num:
                    acc_accuracy += 1

            acc_loss += loss.data[0]
            cnt += 1
        # print("===============test===============")
        # print(acc_loss / cnt)
        print("Loss {}  \tAccuracy {}".format(acc_loss / cnt, acc_accuracy / all_cnt))

        return float(acc_accuracy / all_cnt)


if __name__ == '__main__':
    print("Transfer btw Learnt Code and learnt BoW. "
          "Assume data is Yelp and model is vMF or nor.")
    args = parse_arg()
    print(args)
    # t = Transfer(args)
    # Synthesis data
    bags = []
    # for nor in [True, False]:
    nor = False
    learn = SentClassifier(args, condition=False, nor=nor)
    result = learn.run_train()
    bags.append("nor\t{}\tresult:{}\n".format(nor, result))
    print("nor\t{}\tresult:{}".format(nor, result))
    print(args)
    print("=" * 100)
    for b in bags:
        print(b)
