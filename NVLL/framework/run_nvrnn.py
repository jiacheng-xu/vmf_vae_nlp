import logging
import math
import random
import time

import torch

from NVLL.model.nvrnn import RNNVAE
from NVLL.util.util import schedule, GVar, swap_by_batch, replace_by_batch

random.seed(2018)


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
        dead_cnt = 0
        try:
            for epoch in range(1, self.args.epochs + 1):
                self.args.kl_weight = schedule(epoch)

                epoch_start_time = time.time()

                glob_iter = self.train_epo(self.args, self.model, self.data.train, epoch,
                                           epoch_start_time, glob_iter)

                cur_loss, cur_kl, val_loss = self.evaluate(self.args, self.model,
                                                           self.data.dev)
                val_loss = float(val_loss)
                Runner.log_eval(self.writer, glob_iter, cur_loss, cur_kl, val_loss, False)
                if not best_val_loss or val_loss < best_val_loss:
                    with open(self.args.save_name + ".model", 'wb') as f:
                        torch.save(self.model.state_dict(), f)
                    with open(self.args.save_name + ".args", 'wb') as f:
                        torch.save(self.args, f)
                    best_val_loss = val_loss
                    dead_cnt = 0
                else:
                    dead_cnt += 1
                    self.args.cur_lr /= 1.1
                if dead_cnt == 10:
                    raise KeyboardInterrupt
                    # if epoch == 1 and math.exp(best_val_loss) >= 600:
                    #     raise KeyboardInterrupt
                    # if epoch == 8 and math.exp(best_val_loss) >= 180:
                    #     raise KeyboardInterrupt
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    def end(self):
        # Load the best saved model.
        model = RNNVAE(self.args, self.args.enc_type, len(self.data.dictionary), self.args.emsize,
                       self.args.nhid, self.args.lat_dim, self.args.nlayers,
                       dropout=self.args.dropout, tie_weights=self.args.tied,
                       input_z=self.args.input_z, mix_unk=self.args.mix_unk)
        model.load_state_dict(torch.load(self.args.save_name + '.model'), strict=False)
        model = model.cuda()
        model = model.eval()
        print(model)
        print(self.args)
        cur_loss, cur_kl, test_loss = self.evaluate(self.args, model,
                                                    self.data.test)
        Runner.log_eval(self.writer, None, cur_loss, cur_kl, test_loss, True)
        self.writer.close()

    @staticmethod
    def log_eval(writer, glob_iter, recon_loss, kl_loss, loss, is_test=False):
        # print('=' * 89)
        if is_test:
            print(
                '| End of training | Recon Loss {:5.2f} | KL Loss {:5.2f} | Test Loss {:5.2f} | Test PPL {:8.2f} |'.format(
                    recon_loss, kl_loss, loss, math.exp(loss)))
            if writer is not None:
                writer.add_scalars('test', {'recon_loss': recon_loss, 'kl_loss': kl_loss,
                                            'val_loss': loss,
                                            'ppl': math.exp(loss)
                                            })
        else:
            print('| EVAL | Recon Loss {:5.2f} | KL Loss {:5.2f} | Eval Loss {:5.2f} | Eval PPL {:8.2f} |'.format(
                recon_loss, kl_loss, loss, math.exp(loss)))
            writer.add_scalars('eval', {'recon_loss': recon_loss, 'kl_loss': kl_loss,
                                        'val_loss': loss,
                                        'ppl': math.exp(loss)
                                        }, global_step=glob_iter)
        print('=' * 89)

    @staticmethod
    def log_instant(writer, args, glob_iter, epoch, epoch_start_time,
                    cur_avg_cos, cur_avg_norm, recon_loss
                    , kl_loss, aux_loss,
                    val_loss):
        try:
            print(
                '| epoch {:3d} | time: {:5.2f}s | KL Weight {:5.2f} | AvgCos {:5.2f} | AvgNorm {:5.2f} |Recon Loss {:5.2f} | KL Loss {:5.2f} | Aux '
                'loss: {:5.2f} | Total Loss {:5.2f} | PPL {:8.2f}'.format(
                    epoch, (time.time() - epoch_start_time), args.kl_weight, cur_avg_cos, cur_avg_norm,
                    recon_loss, kl_loss, aux_loss, val_loss, math.exp(val_loss)))
            if writer is not None:
                writer.add_scalars('train', {'lr': args.lr, 'kl_weight': args.kl_weight, 'cur_avg_cos': cur_avg_cos,
                                             'cur_avg_norm': cur_avg_norm, 'recon_loss': recon_loss, 'kl_loss': kl_loss,
                                             'aux_loss': aux_loss,
                                             'val_loss': val_loss,
                                             'ppl': math.exp(val_loss)
                                             }, global_step=glob_iter)
        except OverflowError:
            print('Overflow')

    def train_epo(self, args, model, train_batches, epo, epo_start_time, glob_iter):
        model.train()
        model.FLAG_train = True
        start_time = time.time()

        if self.args.optim == 'sgd':
            self.optim = torch.optim.SGD(model.parameters(), lr=self.args.cur_lr)

        acc_loss = 0
        acc_kl_loss = 0
        acc_aux_loss = 0
        acc_avg_cos = 0
        acc_avg_norm = 0

        batch_cnt = 0
        all_cnt = 0
        cnt = 0

        random.shuffle(train_batches)
        for idx, batch in enumerate(train_batches):
            self.optim.zero_grad()
            seq_len, batch_sz = batch.size()
            feed = self.data.get_feed(batch)

            if self.args.swap > 0.00001:
                feed = swap_by_batch(feed, self.args.swap)
            if self.args.replace > 0.00001:
                feed = replace_by_batch(feed, self.args.replace, self.model.ntoken)

            glob_iter += 1

            target = GVar(batch)

            recon_loss, kld, aux_loss, tup, vecs = model(feed, target)
            total_loss = recon_loss * seq_len + torch.mean(kld) * self.args.kl_weight + torch.mean(
                aux_loss) * args.aux_weight

            total_loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

            self.optim.step()

            acc_loss += recon_loss.data * seq_len * batch_sz
            acc_kl_loss += torch.sum(kld).data
            acc_aux_loss += torch.sum(aux_loss).data
            acc_avg_cos += tup['avg_cos'].data
            acc_avg_norm += tup['avg_norm'].data

            cnt += 1
            batch_cnt += batch_sz
            all_cnt += batch_sz * seq_len
            if idx % args.log_interval == 0 and idx > 0:
                cur_loss = acc_loss[0] / all_cnt
                cur_kl = acc_kl_loss[0] / all_cnt
                # if cur_kl < 0.03:
                #     raise KeyboardInterrupt
                # if cur_kl > 0.7:
                #     raise KeyboardInterrupt
                cur_aux_loss = acc_aux_loss[0] / all_cnt
                cur_avg_cos = acc_avg_cos[0] / cnt
                cur_avg_norm = acc_avg_norm[0] / cnt
                cur_real_loss = cur_loss + cur_kl
                Runner.log_instant(self.writer, self.args, glob_iter, epo, start_time, cur_avg_cos, cur_avg_norm,
                                   cur_loss
                                   , cur_kl, cur_aux_loss,
                                   cur_real_loss)

        return glob_iter

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
