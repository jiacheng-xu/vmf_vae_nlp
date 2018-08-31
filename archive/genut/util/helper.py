import multiprocessing
import pickle as pkl
import sys
import time

import torch
from torch.autograd import Variable as Var
from torch.nn import functional

TRAIN_FLAG = 0
TEST_FLAG = 1
from scipy.linalg import block_diag

stop_words = [',', '.', 'to', 'the', '<\\s>', '<s>']


def schedule(epo):
    # return 400, 90, 1
    if epo <= 4:
        return 250, 60, 0
    elif epo <= 12:
        return 250, 60, 0
    elif epo <= 16:
        return 250, 60, 0
    else:
        return 250, 60, 0


def truncate_mat(length, var):
    if type(var) != list:
        cur_len, batch = var.size()
        if cur_len > length:
            return var[:length, :]
        else:
            return var
    else:
        return [truncate_mat(length, x) for x in var]


def trunc_len(length, var, mask):
    # Var len, batch
    # Mask list, len=batch
    cur_len, batch = var.size()
    batch_ = len(mask)
    if cur_len > length:
        var = var[:length, :]
        mask = [length if m > length else m for m in mask]
        return var, mask
    else:
        return var, mask


def reset_ext_dict_size(bag):
    sz = 0
    if 'replacement' not in bag[0]:
        return 0
    for b in bag:
        sz = max(sz, len(b['replacement']))
    return sz


def show_size(x):
    print(x.data.size())


def mask_translator(mask_list, batch_first=False, is_var=True):
    """
    Given a list of nums, [4, 2,5,1,..], and full seq len, generate a Tensor like
    [111100000, 11000000, ....] Batch*seq
    :param mask_list:
    :param seq_len:
    :return:
    """
    batch_size = len(mask_list)
    seq_len = max(mask_list)
    if batch_first == True:
        mask = np.zeros((batch_size, seq_len))
    else:

        mask = np.zeros((seq_len, batch_size))

    mask = torch.FloatTensor(mask)
    if batch_first:
        for idx, num in enumerate(mask_list):
            mask[idx, :num] = 1
    else:
        for idx, num in enumerate(mask_list):
            mask[:num, idx] = 1
    if is_var:
        mask = Var(mask, requires_grad=False).cuda()
    return mask


def permutated_data(part):
    rand_order = np.random.permutation(len(part))
    part = np.asarray(part)
    permutated_part = part[rand_order]
    return permutated_part


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()  # TODO Check
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)

    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length, use_cuda=True):
    if use_cuda:
        length = Variable(torch.LongTensor(length)).cuda()
        target_flat = target.view(-1, 1).cuda()
    else:
        length = Variable(torch.LongTensor(length))
        target_flat = target.view(-1, 1)
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)

    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def load_prev_model(option, model):
    if option:
        print('Loading %s' % (option))
        # model.load_state_dict(torch.load(option))
        return torch.load(option)
    else:
        return model


def make_batch(opt, part, mode=TRAIN_FLAG):
    """
    Split the whole dataset into batches. First sort the dataset and then split them.
    When using, shuffle the idx to randomly pick up batches.
    :param opt: hp list. feat num >= 1.
    :param part: data with Docs.
    :return: txt_bag is a pack_padded_seq obj.
        abs_bag is a longtensor Var, abs_mask_bag [maxlen of every batch]
    """
    if mode == TRAIN_FLAG:
        batch_size = opt.batch_size
    else:
        batch_size = opt.batch_size_dev
    feat_num = opt.feat_num
    n_doc = len(part)
    part = sorted(part, key=len, reverse=True)

    offset = 0
    last_batch = True
    txt_bag, abs_bag, txt_mask_bag, abs_mask_bag = [], [], [], []
    while last_batch:

        if offset + batch_size >= n_doc:
            last_batch = False
            batch = part[offset:]
            batch_size = len(batch)
        else:
            batch = part[offset:offset + batch_size]
            offset += batch_size

        txt_masks, abs_masks = [], []
        seq_len_txt = min(len(batch[0].tokens_txt), opt.max_len_enc)

        seq_len_abs = 0
        for i in range(len(batch)):
            seq_len_abs = max(seq_len_abs, len(batch[i].tokens_abs))

        if mode == TRAIN_FLAG:
            seq_len_abs = min(seq_len_abs, opt.max_len_dec)  # Trunctuate

        tokens_txt_list = np.zeros((seq_len_txt, batch_size), dtype=int)
        tokens_abs_list = np.zeros((seq_len_abs, batch_size), dtype=int)

        for idx, doc in enumerate(batch):
            txt = np.asarray(doc.tokens_txt)
            abs = np.asarray(doc.tokens_abs)
            # print(txt.shape)
            for col in range(min(txt.shape[0], seq_len_txt)):
                tokens_txt_list[col, idx] = int(txt[col])
            txt_masks.append(col + 1)

            for col in range(min(seq_len_abs, abs.shape[0])):
                tokens_abs_list[col, idx] = int(abs[col])
            abs_masks.append(col + 1)

        tokens_txt_list = torch.LongTensor(tokens_txt_list)

        # For abs, we get
        # abs_mask_mat = np.zeros((seq_len_abs, batch_size), dtype=int)
        # for batch_idx in range(batch_size):
        #     abs_mask_mat[:abs_masks[batch_idx], batch_idx] = 1
        # abs_mask_mat = torch.ByteTensor(abs_mask_mat)

        tokens_abs_list = torch.LongTensor(tokens_abs_list)

        txt_bag.append(tokens_txt_list)
        txt_mask_bag.append(txt_masks)
        abs_bag.append(tokens_abs_list)
        abs_mask_bag.append(abs_masks)

    return txt_bag, abs_bag, txt_mask_bag, abs_mask_bag


def l2_loss(model, scale):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += param ** 2
    reg_loss += scale * reg_loss
    return reg_loss


def single_n_gram(ns, seq):
    ngrams = [[] for _ in range(len(ns))]
    l = len(seq)
    for idx in range(l):
        for i, n in enumerate(ns):
            if idx - n + 1 >= 0:
                tmp = ''
                add = True
                for x in range(idx - n + 1, idx + 1, 1):
                    if str(seq[x]) in stop_words:
                        add = False
                        break
                    tmp += str(seq[x]) + '_'
                if add:
                    ngrams[i].append(tmp)
    return ngrams


def n_gram_list(ns, seqs):
    """

    :param ns: N grams. [2,3,4]
    :param seq: torch.LongTensor. size= (seq_len, batch)
    :return: list, [ [ [a , b, c] [ab, bc, cd,..] ..  ], [], ..  ]
    """
    seqs = seqs.transpose(0, 1).numpy().tolist()
    bags = []
    for i in seqs:
        bags.append(single_n_gram(ns, i))
    return bags


def trim_len(tensor, mask, length):
    seq_len, batch_size = tensor.size()
    batch_size_ = len(mask)
    assert batch_size_ == batch_size
    new_mask = [x if x <= length else length for x in mask]
    if seq_len > length:
        tensor = tensor[:length, :]
    return tensor, new_mask


def one_tensor_replace(tensor, dft_vocab, src_vocab, dft_replacement, src_replacement):
    row, col = tensor.size()
    src_tensor = torch.zeros_like(tensor)
    for c in range(col):
        for r in range(row):

            word_id = tensor[r, c]
            if word_id < len(dft_vocab):
                word_str = dft_vocab.fidx2word(word_id)
            else:
                word_str = dft_replacement.fidx2word(word_id)

            if src_vocab.has_word(word_str):
                src_tensor[r, c] = src_vocab.fword2idx(word_str)
            else:
                src_idx = src_replacement.add_word(word_str)
                src_tensor[r, c] = src_idx
    src_tensor = torch.LongTensor(src_tensor)
    return src_tensor, src_replacement


def single_vocab_replace(inp):
    dft_vocab, one_batch, src_vocab = inp

    dft_txt, dft_abs, msk_txt, msk_abs, dft_replacement, meta = one_batch
    src_replacement = Dict(bias=len(src_vocab))
    src_txt, src_replacement = one_tensor_replace(dft_txt, dft_vocab, src_vocab,
                                                  dft_replacement, src_replacement)

    src_abs, src_replacement = one_tensor_replace(dft_abs, dft_vocab, src_vocab,
                                                  dft_replacement, src_replacement)
    return [src_txt, src_abs, msk_txt, msk_abs, src_replacement, meta]


def vocab_replace(dft_vocab, dft_data, src_vocab):
    """
    Given default_vocab, default_dataset, replace everything in dft_data with new src_vocab.

    :param dft_vocab:
    :param dft_data:
    :param src_vocab:
    :return:
    """
    inp = [[dft_vocab, x, src_vocab] for x in dft_data]

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)

    return_bag = pool.map(single_vocab_replace, inp)
    pool.close()
    pool.join()

    return return_bag


def format_seq(pred_seq):
    if len(pred_seq[-1]) == 1:
        pred_seq[-2] = pred_seq[-2] + pred_seq[-1]
        pred_seq.pop(-1)
    for sid, s in enumerate(pred_seq):
        for wid, w in enumerate(s):
            if w == '<s>':
                pred_seq[sid].remove("<s>")
            elif w == '<\\s>':
                pred_seq[sid].remove("<\\s>")
            elif w == '<\\\s>':
                pred_seq[sid].remove("<\\\s>")
            elif w == '<\\\\s>':
                pred_seq[sid].remove("<\\\\s>")
            elif w == '<\s>':
                pred_seq[sid].remove("<\s>")
    return pred_seq


def demo(txt_seq, pred_seq, gold_seq, p_gens, attn_list, meta):
    tgt_len = len(attn_list)
    print(tgt_len)
    tgt_len_ = len(p_gens)
    print(tgt_len_)
    pred = []
    for x in pred_seq[0]:
        pred += x
    tgt_len__ = len(pred)
    assert tgt_len == tgt_len_ == tgt_len__

    txt = []
    for x in txt_seq[0]:
        txt += x

    gold = []
    for x in gold_seq[0][0]:
        gold += x
    output_txt = ' '.join(txt)
    output_gold = ' '.join(gold)
    output_meta = "%s" % (meta)
    output_pred = []
    for t in range(tgt_len):
        p_gen = p_gens[t][0]
        att = attn_list[t]
        val, idx = torch.topk(att, 1)
        attn_val = val[0]
        attn_txt = txt[idx[0]]
        o = '{:10s} G{:.2f} {:10s} A{:.2f}\n'.format(pred[t], p_gen, attn_txt, attn_val)
        output_pred.append(o)
    output_pred = '\t'.join(output_pred)

    output_string = '\nMeta: %s\nText: %s\nGold: %s\nPred:\n%s\n' % (output_meta, output_txt, output_gold, output_pred)
    return output_string, pred_seq, gold_seq


import numpy as np


def single_scatter(inp):
    tmp_dict = {}
    inp = inp.numpy()
    r_inp = np.flip(inp, 0)
    src_len = inp.size
    maps = np.eye(src_len)

    for s in range(src_len):
        this_id = r_inp[s]
        if tmp_dict.has_key(this_id):
            maps[s, s] = 0
            maps[tmp_dict[this_id], s] = 1
        else:
            tmp_dict[this_id] = s
    final = np.flipud(np.fliplr(maps))
    return final


def prepare_scatter_map(inp_var):
    # Inp src_len, batch
    inp_var = inp_var.transpose(1, 0)
    batch_size, src_len = inp_var.size()
    maps = np.zeros((batch_size, src_len, src_len))
    # s = time.time()
    # Non-par

    for b in range(batch_size):
        batch = inp_var[b]
        rt = single_scatter(batch)
        maps[b] = rt

    # Par
    # sp = time.time()
    # cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=cores)
    # result = pool.map(single_scatter, inp_var)
    # pool.close()
    # pool.join()
    # ep = time.time()
    # print(e-s)
    # print(ep - sp)
    return Var(torch.FloatTensor(maps))


def count_n_gram(inp, n, stop=None):
    assert len(inp) > n
    d = {}
    for i in range(n - 1, len(inp)):
        temp = []
        valid = True
        for j in range(i - n + 1, i + 1):
            if stop is not None:
                if inp[j] in stop:
                    valid = False
                    break
            temp.append(inp[j])
        if valid:
            temp_str = '_'.join([str(t) for t in temp])
            if d.has_key(temp_str):
                pass
            else:
                d[temp_str] = i
    return d

    # x = count_n_gram(['a','b','c','d'],3)
    # print(x)


import hashlib


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def get_gold_label(inp):
    # inp is a batch, with edit distance score
    score = [s.replace('</score', '') for s in inp]
    label_bank = [[] for _ in range(len(score))]
    for i, single_s in enumerate(score):
        ss = single_s.split('<|||>')
        x = [s.split('\t') for s in ss]
        for idx, unit in enumerate(x):
            x[idx] = [int(a) for a in x[idx]]
            label_bank[i].append(x[idx].index(min(x[idx])))
    return label_bank


def generate_trains_batch(raw_txt, inp_var, labels):
    all_bag = []
    for idx in range(len(raw_txt)):
        sample = raw_txt[idx]
        lines = sample.split('<|||>')
        length = []
        for l in lines:
            length.append(len(l.split(' ')))

        # print(length)
        this_inp = [inp_var[0][:, idx], inp_var[1][:, idx], inp_var[2][:, idx]]

        def _align(length, inp_mat):
            splited_mat = []
            current_point = 0
            for n in range(len(length)):
                if current_point + length[n] <= 400:
                    splited_mat.append(inp_mat[current_point:current_point + length[n]].cuda())
                    current_point += length[n]
                    # else:
                    #     splited_mat.append(inp_mat[current_point:])
            return splited_mat

        output = [_align(length, this_inp[0]), _align(length, this_inp[1]), _align(length, this_inp[2])]

        label = labels[idx]
        output_label = [0 for _ in range(len(output[0]))]
        for l in label:
            if l < len(output_label):
                output_label[l] = 1
        assert len(output_label) == len(output[0])
        all_bag.append([output, output_label])
    return all_bag


def align_ori_txt_sent(sample, seq_len):
    lines = sample.split('<|||>')  # list, 'a b c', 'e f g'...
    length_bag = []
    word2sent_idx = []
    word = []

    current_sent_idx = 0
    current_cursor = 0
    for l in lines:
        words = l.split(' ')
        this_length = len(words)
        if current_cursor + this_length <= seq_len:
            word2sent_idx = word2sent_idx + [current_sent_idx] * this_length
            length_bag.append(this_length)
            word += words
            current_cursor += this_length
            current_sent_idx += 1
        else:
            this_length = seq_len - current_cursor
            if this_length == 0:
                break
            word2sent_idx = word2sent_idx + [current_sent_idx] * this_length
            length_bag.append(this_length)
            word += words[:this_length]
            break
    while len(word) < seq_len:
        word.append('<pad>')
        word2sent_idx.append(word2sent_idx[-1])
        length_bag[-1] = length_bag[-1] + 1
    return [word, word2sent_idx, length_bag]


def align_original_txt(ori_txt, seq_len):
    whole_words = []
    whole_sent_idxs = []
    length_bag = []
    for idx in range(len(ori_txt)):
        sample = ori_txt[idx]
        this_word, this_sent_idx, length = align_ori_txt_sent(sample, seq_len)
        whole_words.append(this_word)
        whole_sent_idxs.append(this_sent_idx)
        length_bag.append(length)
    return whole_words, whole_sent_idxs, length_bag


def binary_function(inp, func):
    return 1 if func(inp) else 0


def rec(mat, bid, t, match, seq_len):
    l = 0
    for idx in range(t, seq_len):
        if mat[bid][idx] == match:
            l += 1
            mat[bid][idx] = -1
        else:
            break
    return mat, l


def score_msk_ent(ner_mat, ner_dict):
    seq_len, batch_sz = ner_mat.size()
    assert ner_dict.fword2idx('O') == 0
    # msk = torch.zeros((batch_sz, seq_len))
    ner = ner_mat.transpose(1, 0)
    indicator = torch.gt(ner, 0).int()
    global_bag = []
    for bid in range(batch_sz):
        tmp_bag = []
        for t in range(seq_len):
            if indicator[bid][t] != -1:

                if indicator[bid][t] == 0:
                    indicator, l = rec(indicator, bid, t, indicator[bid][t], seq_len)
                    tmp_bag.append([0, l])
                else:
                    indicator, l = rec(indicator, bid, t, indicator[bid][t], seq_len)
                    tmp_bag.append([1, l])
        global_bag.append(tmp_bag)
    return global_bag


def score_msk_word(mat):
    mat = mat.transpose(1, 0)
    batch_sz, seq_len = mat.size()
    msk = torch.gt(mat, 0).float()

    return msk


def sent_vec2block_diag(inp):
    batch_sz = len(inp)
    seq_len = sum(inp[0])
    mat = np.zeros((batch_sz, seq_len, seq_len))
    for bidx, b in enumerate(inp):
        tmp_bag_for_blocks = np.array([0])

        for l in b:
            tmp_bag_for_blocks = block_diag(tmp_bag_for_blocks, np.ones((l, l)) * (1. / l))

            # tmp_bag_for_blocks.append(np.ones((l,l))*(1./l))
        # tmp_bag_for_blocks = np.asarray(tmp_bag_for_blocks)
        mat[bidx] = tmp_bag_for_blocks[1:, 1:]
    return torch.FloatTensor(mat)


def ent_vec2block_diag(inp, seq_len):
    # inp: batch, seq_len
    batch_sz = len(inp)
    mat = np.zeros((batch_sz, seq_len, seq_len))
    for bidx, b in enumerate(inp):
        tmp_bag_for_blocks = np.array([0])
        for indicator, length in b:
            if indicator == 0:
                tmp_bag_for_blocks = block_diag(tmp_bag_for_blocks, np.zeros((length, length)))
            else:
                tmp_bag_for_blocks = block_diag(tmp_bag_for_blocks, np.ones((length, length)) * (1. / length))
        mat[bidx] = tmp_bag_for_blocks[1:, 1:]

    return torch.FloatTensor(mat)


def read_bin_file(fname):
    with open(fname, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def bigramize(mat_list, src_len, tgt_len):
    batch_sz = len(mat_list)
    tgt_len = min(tgt_len, len(mat_list[0]))
    rt_mat = np.zeros((batch_sz, tgt_len, src_len))
    for bidx in range(batch_sz):
        mat = mat_list[bidx]
        for t in range(tgt_len):
            candidate_positions = mat[t]
            for k in candidate_positions:
                if k < src_len:
                    rt_mat[bidx][t][k] = 1
    return torch.FloatTensor(rt_mat)


# single_scatter(torch.LongTensor([1,20,4,20,20,5,3,2,5,1]))

def concurrent_io(func, files):
    """
    A quick wrap of multiprocessing IO.
    :param func: Function to be paralleled.
    :param files: Files to be loaded.
    :return: output files.
    """
    cores = multiprocessing.cpu_count()
    cores = 3
    pool = multiprocessing.Pool(processes=cores)
    output = pool.map(func, files)
    pool.close()
    pool.join()
    return output


def msk_list_to_mat(inp_list):
    batch_sz = len(inp_list)
    max_len = inp_list[0]
    mask = torch.zeros(max_len, batch_sz).float()

    for idx, l in enumerate(inp_list):
        mask[:l, idx] = 1
    mask = mask.unsqueeze(2)

    return mask
