# coding: utf-8
import argparse
import logging
import math
import time
import os
import torch
import torch.nn as nn

from vae_proto import data
from vae_proto import util

parser = argparse.ArgumentParser(description='PyTorch LSTM Language Model')

parser.add_argument('--data_name', type=str, default='yelp15', help='name of the data corpus')
parser.add_argument('--data_path', type=str, default='../data/yelp15', help='location of the data corpus')

parser.add_argument('--model', type=str, default='vae', help='lstm or vae; VAE or not')
parser.add_argument('--decoder', type=str, default='lstm', help='lstm or bow; Using LSTM or BoW as decoder')

parser.add_argument('--fly', action='store_true', help='Without previous ground truth = inputless decode', default=False)


parser.add_argument('--emsize', type=int, default=300, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')


parser.add_argument('--lr', type=float, default=10,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')

parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, help='evaluation batch size')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')

parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')


parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')


parser.add_argument('--kl_weight', type=float, default=1,
                    help='scalling item for KL')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
fname = 'Model_{}_Fly_{}_Emb{}_Hid{}_nlayer{}_lr{}_drop{}_tied{}_klw{}.log'.format(args.model, args.fly, args.emsize,
                                                                                   args.nhid, args.nlayers, args.lr,
                                                                                   args.dropout, args.tied,
                                                                                   args.kl_weight)
print(fname)

logging.basicConfig(filename=fname, level=logging.INFO)
###############################################################################
# Load data
###############################################################################

# corpus = data.Corpus(args.data)
if 'yelp' in args.data:
    corpus = data.Corpus(args.data,start_idx=1, end_idx=130)
else:
    corpus = data.Corpus(args.data)


eval_batch_size = 10
train_data = util.make_batch(args, corpus.train, args.batch_size)
val_data = util.make_batch(args, corpus.valid, eval_batch_size)
test_data = util.make_batch(args, corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################
ntokens = len(corpus.dictionary)
print('Dict size: %d' % ntokens)

if args.model.lower() == 'lstm':
    from vae_proto import rnn_model

    model = rnn_model.RNNModel("LSTM", ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
elif args.model.lower() == 'vae':
    from vae_proto import vae_model

    model = vae_model.VAEModel("LSTM", ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied,
                               lat_dim=111)
else:
    raise NotImplementedError

args.save = args.model + args.save

# if os.path.isfile(args.save):
#     with open(args.save, 'rb') as f:
#         model = torch.load(f)

if args.cuda:
    model.cuda()
else:
    model.cpu()

logging.info(model)

criterion = nn.CrossEntropyLoss(ignore_index=0)


###############################################################################
# Training code
###############################################################################




def train():
    # Turn on training mode which enables dropout.
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=args.lr)

    acc_loss = 0
    acc_kl_loss = 0
    acc_total_loss = 0

    start_time = time.time()
    ntokens = len(corpus.dictionary)
    # hidden = model.init_hidden(args.batch_size)

    cnt = 0
    glob_iteration = 0
    for batch, i in enumerate(range(0, len(train_data))):
        optim.zero_grad()

        glob_iteration += 1
        data, targets = util.get_batch(args, train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # hidden = repackage_hidden(hidden)
        seq_len, bsz = data.size()

        model.zero_grad()

        # output, hidden = model(data, hidden)
        if args.model == 'lstm':
            if args.fly:
                output, _ = model.forward_decode(args, data, ntokens)
            else:
                output, hidden = model(data)

            loss = criterion(output.view(-1, ntokens), targets)
            total_loss = loss
            total_loss.backward()

        elif args.model == 'vae':
            if args.fly:
                output, _, mu, logvar = model.forward_decode(args, data, ntokens)
            else:
                output, mu, logvar = model(data)

            loss = criterion(output.view(-1, ntokens), targets)
            kld = vae_model.kld(mu, logvar, 1)

            if batch % (args.log_interval / 2) == 0:
                print("RecLoss: %f\tKL: %f" % (loss.data, kld.data))

            total_loss = loss + kld * args.kl_weight
            total_loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)
        optim.step()

        if args.model == 'lstm':
            acc_loss += loss.data * seq_len * bsz
        elif args.model == 'vae':
            acc_loss += loss.data * seq_len * bsz
            acc_kl_loss += kld.data * seq_len * bsz
            acc_total_loss += total_loss.data * seq_len * bsz

        cnt += seq_len * bsz
        if batch % args.log_interval == 0 and batch > 0:
            if args.model == 'lstm':
                cur_loss = acc_loss[0] / cnt
                logging.info(
                    "\t{}\t{}\t{}\t{}".format(epoch, glob_iteration, cur_loss,
                                              math.exp(cur_loss)))

            elif args.model == 'vae':
                cur_loss = acc_loss[0] / cnt
                cur_kl = acc_kl_loss[0] / cnt
                cur_sum = acc_total_loss[0] / cnt

                logging.info(
                    "\t{}\t{}\t{}\t{}\t{}\t{}".format(epoch, glob_iteration, cur_loss, cur_kl, cur_sum,
                                                      math.exp(cur_loss)))

            cnt = 0
            elapsed = time.time() - start_time
            if args.model == 'lstm':
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data), lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))

                acc_loss = 0
            elif args.model == 'vae':

                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | KLW {:5.2f}|'
                      'loss {:5.2f} | KL {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data), lr,
                    elapsed * 1000 / args.log_interval, args.kl_weight, cur_loss, cur_kl, math.exp(cur_loss)))

                acc_loss = 0
                acc_kl_loss = 0
                acc_total_loss = 0
            else:
                raise NotImplementedError

            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        args.kl_weight = util.schedule(epoch)

        epoch_start_time = time.time()
        train()
        val_loss = util.evaluate(args, model, corpus, val_data, criterion)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        with open('valid_PPL_' + fname, 'w') as f:
            f.write("{}\t{}".format(epoch, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 2.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = util.evaluate(args, model, corpus, test_data, criterion)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
