import logging
import math
import random
import time
import numpy
import torch

from NVLL.util.util import schedule, GVar
from NVLL.model.nvrnn import RNNVAE


class Runner():
    def __init__(self, args, model, data, writer):
        self.args = args
        self.data = data
        self.model = model
        self.writer = writer
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.args.cur_lr = self.args.lr
        if args.optim == 'sgd':
            self.optim = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        elif args.optim == 'adam':
            self.optim = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError

    def start(self):
        print("Model {}".format(self.model))
        logging.info("Model {}".format(self.model))
        best_val_loss = None
        glob_iter = 0

        try:
            for epoch in range(1, self.args.epochs + 1):
                self.args.kl_weight = schedule(epoch)

                epoch_start_time = time.time()

                glob_iter = self.train_epo(self.args, self.model, self.data.train, epoch,
                                           epoch_start_time, glob_iter)

                cur_loss, cur_kl, val_loss = self.evaluate(self.args, self.model,
                                                           self.data.dev)
                Runner.log_eval(cur_loss, cur_kl, val_loss, False)

                if not best_val_loss or val_loss < best_val_loss:
                    with open(self.args.save_name + ".model", 'wb') as f:
                        torch.save(self.model.state_dict(), f)
                    with open(self.args.save_name + ".args", 'wb') as f:
                        torch.save(self.args, f)
                    best_val_loss = val_loss
                else:
                    self.args.cur_lr /= 1.2

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    def end(self):
        # Load the best saved model.
        model = RNNVAE(self.args, self.args.enc_type, len(self.data.dictionary), self.args.emsize,
                       self.args.nhid, self.args.lat_dim, self.args.nlayers,
                       dropout=self.args.dropout, tie_weights=self.args.tied)
        model.load_state_dict(torch.load(self.args.save_name + '.model'))
        model = model.cuda()
        print(model)
        print(self.args)
        # with open(self.args.save_name, 'rb') as f:
        #     model = torch.load(f)
        cur_loss, cur_kl, test_loss = self.evaluate(self.args, model,
                                                    self.data.test)
        Runner.log_eval(cur_loss, cur_kl, test_loss, True)
        self.writer.close()

    @staticmethod
    def log_eval(recon_loss, kl_loss, loss, is_test=False):
        # print('=' * 89)
        if is_test:
            print(
                '| End of training | Recon Loss {:5.2f} | KL Loss {:5.2f} | Test Loss {:5.2f} | Test PPL {:8.2f} |'.format(
                    recon_loss, kl_loss, loss, math.exp(loss)))
        else:
            print('| EVAL | Recon Loss {:5.2f} | KL Loss {:5.2f} | Eval Loss {:5.2f} | Eval PPL {:8.2f} |'.format(
                recon_loss, kl_loss, loss, math.exp(loss)))
        print('=' * 89)

    @staticmethod
    def log_instant(writer, args, glob_iter, epoch, epoch_start_time, recon_loss, kl_loss, val_loss):
        try:
            print(
                '| epoch {:3d} | time: {:5.2f}s | KL Weight {:5.2f} | Recon Loss {:5.2f} | KL Loss {:5.2f} | Total Loss {:5.2f} | '
                'PPL {:8.2f}'.format(epoch, (time.time() - epoch_start_time), args.kl_weight,
                                     recon_loss, kl_loss, val_loss, math.exp(val_loss)))
            if writer is not None:
                writer.add_scalars('valid', {'lr': args.lr, 'kl_weight': args.kl_weight,
                                         'val_loss': val_loss,
                                         'ppl': math.exp(val_loss)
                                         }, global_step=glob_iter)
        except OverflowError:
            print('Overflow')
            # with open('Valid_PPL_' + log_name, 'w') as f:
            #     f.write("{}\t{}".format(epoch, math.exp(val_loss)))

    def train_epo(self, args, model, train_batches, epo, epo_start_time, glob_iter):
        model.train()
        model.FLAG_train = True
        start_time = time.time()

        if self.args.optim == 'sgd':
            self.optim = torch.optim.SGD(model.parameters(), lr=self.args.cur_lr)

        acc_loss = 0
        acc_kl_loss = 0
        acc_total_loss = 0

        batch_cnt = 0
        all_cnt = 0
        random.shuffle(train_batches)
        for idx, batch in enumerate(train_batches):
            self.optim.zero_grad()
            seq_len, batch_sz = batch.size()
            feed = self.data.get_feed(batch)

            glob_iter += 1

            # model.zero_grad()

            target = GVar(batch)

            tup, kld, decoded = model(feed, target)

            flatten_decoded = decoded.view(-1, self.model.ntoken)
            flatten_target = target.view(-1)
            loss = self.criterion(flatten_decoded, flatten_target)  # batch_sz * seq, loss
            sum_kld = torch.sum(kld)
            total_loss = loss * seq_len * batch_sz + sum_kld * self.args.kl_weight

            total_loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

            self.optim.step()

            acc_total_loss += loss.data * seq_len * batch_sz + sum_kld.data
            acc_loss += loss.data * seq_len * batch_sz
            acc_kl_loss += sum_kld.data

            batch_cnt += 1
            all_cnt += batch_sz * seq_len
            if idx % args.log_interval == 0 and idx > 0:
                cur_loss = acc_loss[0] / all_cnt
                cur_kl = acc_kl_loss[0] / all_cnt
                # cur_real_loss = acc_real_loss / doc_cnt
                cur_real_loss = acc_total_loss[0] / all_cnt
                Runner.log_instant(self.writer, self.args, glob_iter, epo, start_time, cur_loss
                                   , cur_kl,
                                   cur_real_loss)
                acc_loss, acc_total_loss, acc_kl_loss, all_cnt = 0, 0, 0, 0

        return glob_iter

    def evaluate(self, args, model, dev_batches):

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
