import logging
import math
import random
import time
from NVLL.util.util import anneal_list
import torch

from NVLL.model.nvrnn import RNNVAE
from NVLL.util.gpu_flag import device
from NVLL.util.util import schedule, GVar, swap_by_batch, replace_by_batch

random.seed(2018)
import os


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
        self.glob_iter = 0
        self.dead_cnt = 0
        self.best_val_loss = None

    def start(self):
        print("Model {}".format(self.model))
        logging.info("Model {}".format(self.model))
        try:
            for epoch in range(1, self.args.epochs + 1):
                self.args.kl_weight = schedule(epoch, self.args.anneal)

                epoch_start_time = time.time()

                self.train_epo(self.args, self.model, self.data.train, epoch,
                               epoch_start_time, self.glob_iter)

                cur_loss, cur_kl, val_loss = self.evaluate(self.args, self.model,
                                                           self.data.dev)
                val_loss = float(val_loss)
                Runner.log_eval(self.writer, self.glob_iter, cur_loss, cur_kl, val_loss, False)
                print(self.args.save_name)
                if not self.best_val_loss or val_loss < self.best_val_loss:
                    with open(self.args.save_name + ".model", 'wb') as f:
                        torch.save(self.model.state_dict(), f)
                    with open(self.args.save_name + ".args", 'wb') as f:
                        torch.save(self.args, f)
                    self.best_val_loss = val_loss
                    self.dead_cnt = 0
                else:
                    self.dead_cnt += 1
                    self.args.cur_lr /= 1.2
                if self.dead_cnt == 15:
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
                       input_z=self.args.input_z, mix_unk=self.args.mix_unk,
                       condition=(self.args.cd_bit or self.args.cd_bow),
                       input_cd_bow=self.args.cd_bow, input_cd_bit=self.args.cd_bit)
        model.load_state_dict(torch.load(self.args.save_name + '.model'), strict=False)
        model.to(device)
        # if torch.cuda.is_available() and GPU_FLAG:
        #     model = model.cuda()
        model = model.eval()
        print(model)
        print(self.args)
        # print("Anneal Type: {}".format(anneal_list[self.args.anneal]))
        train_loss, train_kl, train_total_loss = self.evaluate(self.args, model,
                                                               self.data.train)

        cur_loss, cur_kl, test_loss = self.evaluate(self.args, model,
                                                    self.data.test)
        Runner.log_eval(self.writer, None, cur_loss, cur_kl, test_loss, True)

        os.rename(self.args.save_name + '.model', self.args.save_name + '_' + str(test_loss) + '.model')
        os.rename(self.args.save_name + '.args', self.args.save_name + '_' + str(test_loss) + '.args')

        # Write result to board
        self.write_board(self.args, train_loss, train_kl,
                         train_total_loss, cur_loss, cur_kl, test_loss)
        self.writer.close()

    @staticmethod
    def write_board(args, train_loss, train_kl, train_total_loss, cur_loss, cur_kl, test_loss):
        from datetime import datetime
        with open(os.path.join(args.exp_path, args.board), 'a') as fd:
            part_id = str(datetime.utcnow()) + "\t"
            for k, v in vars(args).items():
                part_id += str(k) + ":\t" + str(v) + "\t"
            part_loss = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                train_loss, train_kl, train_total_loss, math.exp(train_total_loss),
                cur_loss, cur_kl, test_loss, math.exp(test_loss))
            print(part_id + part_loss)
            fd.write(part_id + part_loss)

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
        # if kl_loss < 0.02 and args.dist == 'vmf':
        #     raise KeyboardInterrupt
        # if kl_loss >1.0 and args.dist == 'vmf':
        #     raise KeyboardInterrupt
        # if kl_loss > 0.4 and args.mix_unk < 0.01 and args.dist == 'vmf':
        #     raise KeyboardInterrupt
        # if kl_loss < 0.05 and args.mix_unk > 0.99 and args.dist == 'vmf':
        #     raise KeyboardInterrupt
        try:
            print(
                '| epoch {:3d} | time: {:5.2f}s | Iter: {} | KL Weight {:5.2f} | AvgCos {:5.2f} | AvgNorm {:5.2f} |Recon Loss {:5.2f} | KL Loss {:5.2f} | Aux '
                'loss: {:5.2f} | Total Loss {:5.2f} | PPL {:8.2f}'.format(
                    epoch, (time.time() - epoch_start_time), glob_iter, args.kl_weight, cur_avg_cos, cur_avg_norm,
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

            if self.args.swap > 0.00001:
                feed = swap_by_batch(feed, self.args.swap)
            if self.args.replace > 0.00001:
                feed = replace_by_batch(feed, self.args.replace, self.model.ntoken)

            self.glob_iter += 1

            target = GVar(batch)

            recon_loss, kld, aux_loss, tup, vecs, _ = model(feed, target, bit)
            total_loss = recon_loss * seq_len + torch.mean(kld) * self.args.kl_weight + torch.mean(
                aux_loss) * args.aux_weight

            total_loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)        # Upgrade to pytorch 0.4.1
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip, norm_type=2)

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
                cur_loss = acc_loss.item() / all_cnt
                cur_kl = acc_kl_loss.item() / all_cnt
                # if cur_kl < 0.03:
                #     raise KeyboardInterrupt
                # if cur_kl > 0.7:
                #     raise KeyboardInterrupt
                cur_aux_loss = acc_aux_loss.item() / all_cnt
                cur_avg_cos = acc_avg_cos.item() / cnt
                cur_avg_norm = acc_avg_norm.item() / cnt
                cur_real_loss = cur_loss + cur_kl
                Runner.log_instant(self.writer, self.args, self.glob_iter, epo, start_time, cur_avg_cos, cur_avg_norm,
                                   cur_loss
                                   , cur_kl, cur_aux_loss,
                                   cur_real_loss)

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

            recon_loss, kld, aux_loss, tup, vecs, _ = model(feed, target, bit)

            acc_loss += recon_loss.data * seq_len * batch_sz
            acc_kl_loss += torch.sum(kld).data
            acc_aux_loss += torch.sum(aux_loss).data
            acc_avg_cos += tup['avg_cos'].data
            acc_avg_norm += tup['avg_norm'].data
            cnt += 1
            batch_cnt += batch_sz
            all_cnt += batch_sz * seq_len

        cur_loss = acc_loss.item() / all_cnt
        cur_kl = acc_kl_loss.item() / all_cnt
        cur_aux_loss = acc_aux_loss.item() / all_cnt
        cur_avg_cos = acc_avg_cos.item() / cnt
        cur_avg_norm = acc_avg_norm.item() / cnt
        cur_real_loss = cur_loss + cur_kl

        # Runner.log_eval(print_ppl)
        # print('loss {:5.2f} | KL {:5.2f} | ppl {:8.2f}'.format(            cur_loss, cur_kl, math.exp(print_ppl)))
        return cur_loss, cur_kl, cur_real_loss
