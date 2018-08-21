import math
import os
import random

import numpy as np
import torch
from NVLL.util.gpu_flag import device


class NewsCorpus(object):
    def __init__(self, path):
        self.test, self.test_cnt = self.read_data(os.path.join(path, 'test.feat'))
        self.train, self.train_cnt = self.read_data(os.path.join(path, 'train.feat'))
        l = list(range(len(self.test)))
        random.shuffle(l)
        l = l[:500]
        self.dev = []
        self.dev_cnt = []
        for i in l:
            self.dev.append(self.test[i])
            self.dev_cnt.append(self.test_cnt[i])

    def read_data(self, path_file):
        _id = 0
        idx = []
        data = []
        word_count = []
        fin = open(path_file)
        while True:
            line = fin.readline()
            if not line:
                break
            id_freqs = line.split()
            doc = {}
            count = 0
            for id_freq in id_freqs[1:]:
                items = id_freq.split(':')
                # python starts from 0
                doc[int(items[0]) - 1] = int(items[1])
                count += int(items[1])
            if count > 0:
                idx.append(_id)
                _id += 1
                data.append(doc)
                word_count.append(count)
        fin.close()
        # sorted_idx = sorted(idx, key=lambda sample: word_count[sample], reverse=True)
        # new_data = []
        # new_count = []
        # for i, this_id in enumerate(sorted_idx):
        #     new_data.append(data[this_id])
        #     new_count.append(word_count[this_id])
        return data, word_count


def data_set(data_url):
    """process data input."""
    data = []
    word_count = []
    fin = open(data_url)
    while True:
        line = fin.readline()
        if not line:
            break
        id_freqs = line.split()
        doc = {}
        count = 0
        for id_freq in id_freqs[1:]:
            items = id_freq.split(':')
            # python starts from 0
            doc[int(items[0]) - 1] = int(items[1])
            count += int(items[1])
        if count > 0:
            data.append(doc)
            word_count.append(count)
    fin.close()
    return data, word_count


def create_batches(data_size, batch_size, shuffle=True):
    """create index by batches."""
    batches = []
    ids = list(range(data_size))
    if shuffle:
        random.shuffle(ids)
    for i in range(int(data_size / batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append(ids[start:end])
    # the batch of which the length is less than batch_size
    rest = data_size % batch_size
    if rest > 0:
        # batches.append(list(ids[-rest:]) + [-1] * (batch_size - rest))  # -1 as padding
        batches.append(list(ids[-rest:]))  # -1 as padding
    return batches


def fetch_data(data, count, idx_batch, vocab_size):
    """fetch input data by batch."""
    batch_size = len(idx_batch)
    data_batch = np.zeros((batch_size, vocab_size))
    count_batch = []
    mask = np.zeros(batch_size)
    indices = []
    values = []
    for i, doc_id in enumerate(idx_batch):
        if doc_id != -1:
            for word_id, freq in data[doc_id].items():
                data_batch[i, word_id] = freq
            count_batch.append(count[doc_id])
            mask[i] = 1.0
        else:
            count_batch.append(0)
    return data_batch, count_batch, mask


def schedule(epo, eval=False):
    return float(torch.sigmoid(torch.ones(1) * (epo - 5)))


def variable_parser(var_list, prefix):
    """return a subset of the all_variables by prefix."""
    ret_list = []
    for var in var_list:
        varname = var.name
        varprefix = varname.split('/')[0]
        if varprefix == prefix:
            ret_list.append(var)
    return ret_list


#
#
# def linear(inputs,
#            output_size,
#            no_bias=False,
#            bias_start_zero=False,
#            matrix_start_zero=False,
#            scope=None):
#     """Define a linear connection."""
#     with tf.variable_scope(scope or 'Linear'):
#         if matrix_start_zero:
#             matrix_initializer = tf.constant_initializer(0)
#         else:
#             matrix_initializer = None
#         if bias_start_zero:
#             bias_initializer = tf.constant_initializer(0)
#         else:
#             bias_initializer = None
#         input_size = inputs.get_shape()[1].value
#         matrix = tf.get_variable('Matrix', [input_size, output_size],
#                                  initializer=matrix_initializer)
#         bias_term = tf.get_variable('Bias', [output_size],
#                                     initializer=bias_initializer)
#         output = tf.matmul(inputs, matrix)
#         if not no_bias:
#             output = output + bias_term
#     return output
#
#
# def mlp(inputs,
#         mlp_hidden=[],
#         mlp_nonlinearity=tf.nn.tanh,
#         scope=None):
#     """Define an MLP."""
#     with tf.variable_scope(scope or 'Linear'):
#         mlp_layer = len(mlp_hidden)
#         res = inputs
#         for l in range(mlp_layer):
#             res = mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l' + str(l)))
#         return res
#
#
import time


def evaluate(args, model, corpus_dev, corpus_dev_cnt, dev_batches):
    # Turn on training mode which enables dropout.
    model.eval()

    acc_loss = 0
    acc_kl_loss = 0
    acc_real_ppl = 0
    word_cnt = 0
    doc_cnt = 0
    start_time = time.time()
    ntokens = 2000

    for idx, batch in enumerate(dev_batches):
        data_batch, count_batch, mask = fetch_data(
            corpus_dev, corpus_dev_cnt, batch, ntokens)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # hidden = repackage_hidden(hidden)

        data_batch = torch.FloatTensor(data_batch).to(device)
        mask = torch.FloatTensor(mask).to(device)

        recon_loss, kld, _ = model(data_batch, mask)
        count_batch = torch.FloatTensor(count_batch).to(device)
        real_ppl = torch.div((recon_loss + kld).data, count_batch) * mask.data

        # remove nan
        for n in real_ppl:
            if n == n:
                acc_real_ppl += n
        # acc_real_ppl += torch.sum(real_ppl)

        acc_loss += torch.sum(recon_loss).data  #
        # acc_kl_loss += kld.data * torch.sum(mask.data)
        acc_kl_loss += torch.sum(kld.data * torch.sum(mask.data))
        count_batch = count_batch + 1e-12
        word_cnt += torch.sum(count_batch)
        doc_cnt += torch.sum(mask.data)

    # word ppl
    cur_loss = acc_loss[0] / word_cnt  # word loss
    cur_kl = acc_kl_loss / doc_cnt
    print_ppl = acc_real_ppl / doc_cnt

    elapsed = time.time() - start_time

    print('loss {:5.2f} | KL {:5.2f} | ppl {:8.2f}'.format(

        cur_loss, cur_kl, np.exp(print_ppl)))
    return print_ppl
