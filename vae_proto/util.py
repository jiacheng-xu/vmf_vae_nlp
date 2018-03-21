
import torch
import torch.nn as nn
from torch.autograd import Variable
import random


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(args, source, i, evaluation=False):
    # seq_len = min(args.bptt, len(source) - 1 - i)

    data_patch = source[i]
    bsz = data_patch.size()[1]

    sos = torch.LongTensor(1, bsz).fill_(1)
    if args.cuda:
        sos = sos.cuda()

    input_data = Variable(torch.cat((sos, data_patch[:-1])), volatile=evaluation)
    # print(input_data.size())
    # exit()
    # data = Variable(source[i:i + seq_len], volatile=evaluation)
    target = Variable(data_patch.view(-1))
    return input_data, target


def make_single_batch(args, buff):
    max_len = len(buff[0][0])
    bsz = len(buff)
    batch = torch.LongTensor(max_len, bsz).fill_(0)
    for n in range(bsz):
        seq_len = len(buff[n][0])
        for t in range(seq_len):
            batch[t, n] = buff[n][0][t]
    if args.cuda:
        batch = batch.cuda()
    return batch


def make_batch(args, data_bag, bsz,shuffle=True):
    all_batched_data = []
    total_num = len(data_bag)
    n_batch = total_num // bsz
    if total_num % bsz != 0:
        n_batch += 1

    buffer = []
    for idx, data_patch in enumerate(data_bag):
        buffer.append(data_patch)
        if len(buffer) == bsz:
            result = make_single_batch(args, buffer)
            all_batched_data.append(result)
            buffer = []
    if buffer != []:
        result = make_single_batch(args, buffer)
        all_batched_data.append(result)

    if shuffle:
        random.shuffle(all_batched_data)

    return all_batched_data


def evaluate(args, model, corpus, data_source, crit=nn.CrossEntropyLoss(ignore_index=0)):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    cnt = 0

    for batch, i in enumerate(range(0, len(data_source))):
        data, targets = get_batch(args, data_source, i, evaluation=True)
        seq_len, bsz = data.size()
        output =  model(data)[0]
        output_flat = output.view(-1, ntokens)
        total_loss += seq_len * bsz * crit(output_flat, targets).data
        cnt += seq_len * bsz
    return total_loss[0] / cnt

def decode_inputless(args, model, corpus, data_source):
    model.eval()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, len(data_source))):
        data, targets = get_batch(args, data_source, i, evaluation=True)
        seq_len, bsz = data.size()
        model.forward_decode(data, ntokens)