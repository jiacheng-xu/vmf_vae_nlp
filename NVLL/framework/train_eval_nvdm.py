"""
Runner is responsible for basically everything bef, during and after training and testing.
"""
import logging
import math
import os
import random
import time

import torch
from NVLL.util.gpu_flag import device
from NVLL.data.ng import DataNg

# from NVLL.util.util import schedule, GVar, maybe_cuda
from NVLL.util.util import schedule, GVar

random.seed(2018)


class Runner():
    def __init__(self, args, model, data, writer):
        self.args = args
        self.data = data
        self.model = model
        self.writer = writer
        self.args.cur_lr = self.args.lr
        if args.optim == 'sgd':
            self.optim = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        elif args.optim == 'adam':
            self.optim = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError
        self.dead_cnt = 0
        self.glob_iter = 0
        self.best_val_loss = None

    def start(self):
        print("Model {}".format(self.model))
        logging.info("Model {}".format(self.model))

        try:
            for epoch in range(1, self.args.epochs + 1):
                # self.args.kl_weight = schedule(epoch)

                epoch_start_time = time.time()

                self.data.set_train_batches(self.args)

                self.train_epo(self.args, self.model, self.data.train_batches, epoch,
                               epoch_start_time)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    @staticmethod
    def write_board(args, cur_loss, cur_kl, test_loss):
        from datetime import datetime
        with open(os.path.join(args.exp_path, args.board), 'a') as fd:
            part_id = str(datetime.utcnow()) + "\t"
            for k, v in vars(args).items():
                part_id += str(k) + ":\t" + str(v) + "\t"
            part_loss = "{}\t{}\t{}\t{}\n".format(

                cur_loss, cur_kl, test_loss, math.exp(test_loss))
            print(part_id + part_loss)
            fd.write(part_id + part_loss)

    def end(self):
        # Load the best saved model.
        if self.args.data_name == '20ng':
            from NVLL.model.nvdm import BowVAE
        elif self.args.data_name == 'rcv':
            from NVLL.model.nvdm import BowVAE
        else:
            raise NotImplementedError
        model = BowVAE(self.args, vocab_size=self.data.vocab_size, n_hidden=self.args.nhid,
                       n_lat=self.args.lat_dim,
                       n_sample=3, dist=self.args.dist)
        model.load_state_dict(torch.load(self.args.save_name + '.model'),
                              strict=False)
        model.to(device)
        # if torch.cuda.is_available() and GPU_FLAG:
        #     model = model.cuda()
        print(model)
        print(self.args)
        model = model.eval()

        cur_loss, cur_kl, test_loss = self.evaluate(self.args, model,
                                                    self.data.test[0], self.data.test[1], self.data.test_batches)
        Runner.log_eval(self.writer, None, cur_loss, cur_kl, test_loss, True)
        cur_loss = cur_loss.item()
        cur_kl = cur_kl.item()
        test_loss = test_loss.item()
        os.rename(self.args.save_name + '.model', self.args.save_name + '_' + str(test_loss) + '.model')
        os.rename(self.args.save_name + '.args', self.args.save_name + '_' + str(test_loss) + '.args')
        self.write_board(self.args, cur_loss, cur_kl, test_loss)
        self.writer.close()

    @staticmethod
    def log_eval(writer, glob_iter, recon_loss, kl_loss, loss, is_test=False):
        recon_loss = recon_loss.item()
        kl_loss = kl_loss.item()
        loss = loss.item()
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
            print(
                '| EVAL | Step: {} | Recon Loss {:5.2f} | KL Loss {:5.2f} | Eval Loss {:5.2f} | Eval PPL {:8.2f} |'.format(
                    glob_iter, recon_loss, kl_loss, loss, math.exp(loss)))
            writer.add_scalars('eval', {'recon_loss': recon_loss, 'kl_loss': kl_loss,
                                        'val_loss': loss,
                                        'ppl': math.exp(loss)
                                        }, global_step=glob_iter)
        print('=' * 89)

    @staticmethod
    def log_instant(writer, args, glob_iter, epoch, epoch_start_time, cur_avg_cos, cur_avg_norm,
                    recon_loss, kl_loss, aux_loss, val_loss):
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

    def train_epo(self, args, model, train_batches, epo, epo_start_time):
        model.train()
        start_time = time.time()

        if self.args.optim == 'sgd':
            self.optim = torch.optim.SGD(model.parameters(), lr=self.args.cur_lr)
        else:
            raise NotImplementedError
        acc_loss = 0
        acc_kl_loss = 0
        acc_aux_loss = 0
        acc_avg_cos = 0
        acc_avg_norm = 0
        # acc_real_loss = 0

        word_cnt = 0
        doc_cnt = 0
        cnt = 0
        random.shuffle(train_batches)
        for idx, batch in enumerate(train_batches):
            self.optim.zero_grad()

            self.glob_iter += 1
            data_batch, count_batch = DataNg.fetch_data(
                self.data.train[0], self.data.train[1], batch, self.data.vocab_size)

            model.zero_grad()

            data_batch = GVar(torch.FloatTensor(data_batch))

            recon_loss, kld, aux_loss, tup, vecs = model(data_batch)
            # print("Recon: {}\t KL: {}".format(recon_loss,kld))
            # total_loss = torch.mean(recon_loss + kld * args.kl_weight)
            total_loss = torch.mean(recon_loss + kld * args.kl_weight + aux_loss * args.aux_weight)
            total_loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            self.optim.step()

            count_batch = GVar(torch.FloatTensor(count_batch))
            doc_num = len(count_batch)

            # real_loss = torch.div((recon_loss + kld).data, count_batch)
            # acc_real_loss += torch.sum(real_loss)

            acc_loss += torch.sum(recon_loss).item()
            acc_kl_loss += torch.sum(kld).item()
            acc_aux_loss += torch.sum(aux_loss).item()
            acc_avg_cos += tup['avg_cos'].item()
            acc_avg_norm += tup['avg_norm'].item()
            cnt += 1

            count_batch = count_batch + 1e-12
            word_cnt += torch.sum(count_batch).item()
            doc_cnt += doc_num

            if idx % args.log_interval == 0 and idx > 0:
                cur_loss = acc_loss / word_cnt  # word loss
                cur_kl = acc_kl_loss / word_cnt
                cur_aux_loss = acc_aux_loss / word_cnt
                cur_avg_cos = acc_avg_cos / cnt
                cur_avg_norm = acc_avg_norm / cnt
                # cur_real_loss = acc_real_loss / doc_cnt
                cur_real_loss = cur_loss + cur_kl

                # if cur_kl < 0.14 or cur_kl > 1.2:
                #     raise KeyboardInterrupt

                Runner.log_instant(self.writer, self.args, self.glob_iter, epo, start_time,
                                   cur_avg_cos, cur_avg_norm, cur_loss
                                   , cur_kl, cur_aux_loss,
                                   cur_real_loss)
                acc_loss = 0
                acc_kl_loss = 0
                acc_aux_loss = 0
                acc_avg_cos = 0
                acc_avg_norm = 0
                word_cnt = 0
                doc_cnt = 0
                cnt = 0
            if idx % (3 * args.log_interval) == 0 and idx > 0:
                with torch.no_grad():
                    self.eval_interface()

    def eval_interface(self):

        cur_loss, cur_kl, val_loss = self.evaluate(self.args, self.model,
                                                   self.data.dev[0], self.data.dev[1], self.data.dev_batches)
        Runner.log_eval(self.writer, self.glob_iter, cur_loss, cur_kl, val_loss, False)
        print(self.args.save_name)
        val_loss = val_loss.item()
        if not self.best_val_loss or val_loss < self.best_val_loss:
            with open(self.args.save_name + ".model", 'wb') as f:
                torch.save(self.model.state_dict(), f)
            with open(self.args.save_name + ".args", 'wb') as f:
                torch.save(self.args, f)
            self.best_val_loss = val_loss
            self.dead_cnt = 0
        else:
            self.dead_cnt += 1
            self.args.cur_lr /= 1.1
        # if self.glob_iter > 1000:
        #     raise KeyboardInterrupt

        if self.dead_cnt == 15:
            raise KeyboardInterrupt

    def evaluate(self, args, model, corpus_dev, corpus_dev_cnt, dev_batches):

        # Turn on training mode which enables dropout.
        model.eval()

        acc_loss = 0
        acc_kl_loss = 0
        acc_real_loss = 0
        word_cnt = 0
        doc_cnt = 0
        start_time = time.time()
        ntokens = self.data.vocab_size

        for idx, batch in enumerate(dev_batches):
            data_batch, count_batch = self.data.fetch_data(
                corpus_dev, corpus_dev_cnt, batch, ntokens)

            data_batch = GVar(torch.FloatTensor(data_batch))

            recon_loss, kld, aux_loss, tup, vecs = model(data_batch)

            count_batch = GVar(torch.FloatTensor(count_batch))
            # real_loss = torch.div((recon_loss + kld).data, count_batch)
            doc_num = len(count_batch)
            # remove nan
            # for n in real_loss:
            #     if n == n:
            #         acc_real_loss += n
            # acc_real_ppl += torch.sum(real_ppl)

            acc_loss += torch.sum(recon_loss).item()  #
            acc_kl_loss += torch.sum(kld).item()
            count_batch = count_batch + 1e-12

            word_cnt += torch.sum(count_batch)
            doc_cnt += doc_num

        # word ppl
        cur_loss = acc_loss / word_cnt  # word loss
        cur_kl = acc_kl_loss / word_cnt
        # cur_real_loss = acc_real_loss / doc_cnt
        cur_real_loss = cur_loss + cur_kl
        elapsed = time.time() - start_time

        # Runner.log_eval(print_ppl)
        # print('loss {:5.2f} | KL {:5.2f} | ppl {:8.2f}'.format(            cur_loss, cur_kl, math.exp(print_ppl)))
        return cur_loss, cur_kl, cur_real_loss
