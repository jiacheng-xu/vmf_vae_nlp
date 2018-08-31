# coding: utf-8
import argparse
import logging
import math
import os
import time
import torch
import numpy as np

from tensorboardX import SummaryWriter

from archive import nvdm as util
from archive.nvdm import BowVAE

parser = argparse.ArgumentParser(description='PyTorch VAE LSTM Language Model')

parser.add_argument('--data_name', type=str, default='20news', help='name of the data corpus')
parser.add_argument('--data_path', type=str, default='../data/20news', help='location of the data corpus')

parser.add_argument('--distribution', type=str, default='nor', help='nor or vmf')

parser.add_argument('--kappa', type=float, default=5)

parser.add_argument('--emsize', type=int, default=800, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=800, help='number of hidden units per layer')
parser.add_argument('--lat_dim', type=int, default=800, help='dim of latent vec z')

parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')

parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, help='evaluation batch size')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')

parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')

parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')

parser.add_argument('--kl_weight', type=float, default=1,
                    help='scaling item for KL')

parser.add_argument('--load', type=str, default=None, help='restoring previous model')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

args.save_name = 'Data{}_Dist{}_Emb{}_Hid{}_lat{}_lr{}_drop{}'.format(
    args.data_name, str(args.dist),
    args.emsize,
    args.nhid, args.lat_dim, args.lr,
    args.dropout)
writer = SummaryWriter(log_dir='exps/' + args.save_name)

log_name = args.save_name + '.log'
logging.basicConfig(filename=log_name, level=logging.INFO)
###############################################################################
# Load data
###############################################################################

# corpus = data.Corpus(args.data)
if '20news' in args.data_name:
    corpus = util.NewsCorpus(args.data_path)
else:
    raise NotImplementedError

test_batches = util.create_batches(len(corpus.test), args.eval_batch_size, shuffle=True)
dev_batches = util.create_batches(len(corpus.dev), args.eval_batch_size, shuffle=True)

###############################################################################
# Build the model
###############################################################################

model = BowVAE(vocab_size=2000, n_hidden=args.nhid, n_lat=args.lat_dim,
               n_sample=5, batch_size=args.batch_size, non_linearity='Tanh', dist=args.dist)

print("Model {}".format(model))
logging.info("Model {}".format(model))

if args.load != None:
    if os.path.isfile(args.load):
        with open(args.load, 'rb') as f:
            model = torch.load(f)
        logging.info("Successfully load previous model! {}".format(args.load))

if args.cuda:
    model.cuda()
else:
    model.cpu()

logging.info(model)


###############################################################################
# Training code
###############################################################################

def train(train_batches, glob_iteration):
    # Turn on training mode which enables dropout.
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    acc_loss = 0
    acc_kl_loss = 0
    word_cnt = 0
    doc_cnt = 0
    acc_real_ppl = 0
    start_time = time.time()
    ntokens = 2000

    for idx, batch in enumerate(train_batches):
        optim.zero_grad()

        glob_iteration += 1
        data_batch, count_batch, mask = util.fetch_data(
            corpus.train, corpus.train_cnt, batch, ntokens)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # hidden = repackage_hidden(hidden)

        model.zero_grad()

        data_batch = torch.autograd.Variable(torch.FloatTensor(data_batch))
        mask = torch.autograd.Variable(torch.FloatTensor(mask))
        if args.cuda:
            data_batch = data_batch.cuda()
            mask = mask.cuda()

        recon_loss, kld, _ = model(data_batch, mask)

        if idx % (args.log_interval / 2) == 0:
            print("RecLoss: %f\tKL: %f" % (torch.mean(recon_loss).data, torch.mean(kld).data))

        total_loss = torch.mean(recon_loss + kld * args.kl_weight)
        total_loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        optim.step()
        count_batch = torch.FloatTensor(count_batch).cuda()
        real_ppl = torch.div((recon_loss + kld).data, count_batch) * mask.data
        acc_real_ppl += torch.sum(real_ppl)

        acc_loss += torch.sum(recon_loss).data  #
        # print(kld.size(), mask.size())
        acc_kl_loss += torch.sum(kld.data * torch.sum(mask.data))

        count_batch = count_batch + 1e-12
        word_cnt += torch.sum(count_batch)
        doc_cnt += torch.sum(mask.data)

        if idx % args.log_interval == 0 and idx > 0:
            # word ppl
            cur_loss = acc_loss[0] / word_cnt  # word loss
            cur_kl = acc_kl_loss / doc_cnt
            print_ppl = acc_real_ppl / doc_cnt
            logging.info(
                "\t{}\t{}\t{}\t{}\t{}\t{}".format(epoch, glob_iteration, cur_loss, cur_kl, print_ppl,
                                                  math.exp(print_ppl)))
            writer.add_scalars('train', {'lr': args.lr, 'kl_weight': args.kl_weight,
                                         'cur_loss': cur_loss, 'cur_kl': cur_kl,
                                         'cur_sum': print_ppl, 'ppl': math.exp(print_ppl)
                                         }, global_step=glob_iteration)

            elapsed = time.time() - start_time

            print('| epoch {:3d} | {:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | KLW {:5.2f}|'
                  'loss {:5.2f} | KL {:5.2f} | ppl {:8.2f}'.format(
                epoch, idx, lr,
                elapsed * 1000 / args.log_interval, args.kl_weight, cur_loss, cur_kl,
                np.exp(print_ppl)))

            word_cnt = 0
            doc_cnt = 0
            acc_loss = 0
            acc_kl_loss = 0
            acc_real_ppl = 0
            start_time = time.time()

    return glob_iteration


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    glob_iter = 0
    for epoch in range(1, args.epochs + 1):
        args.kl_weight = util.schedule(epoch)

        epoch_start_time = time.time()
        train_batches = util.create_batches(len(corpus.train), args.batch_size, shuffle=True)
        glob_iter = train(train_batches, glob_iter)
        val_loss = util.evaluate(args, model, corpus.dev, corpus.dev_cnt, dev_batches)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        writer.add_scalars('valid', {'lr': args.lr, 'kl_weight': args.kl_weight,
                                     'val_loss': val_loss,
                                     'ppl': math.exp(val_loss)
                                     }, global_step=glob_iter)
        print('-' * 89)
        with open('Valid_PPL_' + log_name, 'w') as f:
            f.write("{}\t{}".format(epoch, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save_name, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 1.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save_name, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = util.evaluate(args, model, corpus.test, corpus.test_cnt, test_batches)
print('=' * 89)
print('| End of training | Test Loss {:5.2f} | Test PPL {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

writer.close()
