import random
import os
import spacy
import nltk


def preprocess_stage_1(path, train_file, test_file):
    """
    Goal: generate train.txt valid.txt test.txt
    format: [classlabel]\t[sentences....]
    :param path:
    :param train_file:
    :param test_file:
    :return:
    """
    # Sample 100K from train and 10K form valid and test
    # Train 650000  Test 50000

    train_idxs = random.sample(range(650000), 100000)
    valid_test_idxs = random.sample(range(50000), 20000)
    valid_idxs = valid_test_idxs[:10000]
    test_idxs = valid_test_idxs[10000:]

    def _preprocess_line(line):
        pred_comma = line.find(',')
        expected_comma = 3
        assert pred_comma == expected_comma
        label, sent = line[1:expected_comma - 1], line[expected_comma + 2:-1]
        return label + '\t' + sent

    def _read_file(path, which_file, keep_num):
        with open('/'.join([path, which_file]), 'r') as tr:
            lines = tr.read().splitlines()

        random.shuffle(lines)
        lines = lines[:keep_num]

        bag = []
        for idx, line in enumerate(lines):
            new_line = _preprocess_line(line)
            bag.append(new_line)
        return bag

    trains = _read_file(path, train_file, 100000)
    valid_and_test = _read_file(path, test_file, 20000)

    with open(os.path.join(path, 'train.txt'), 'w') as fd:
        fd.write('\n'.join(trains))

    valids = valid_and_test[:10000]
    with open(os.path.join(path, 'valid.txt'), 'w') as fd:
        fd.write('\n'.join(valids))

    tests = valid_and_test[10000:]
    with open(os.path.join(path, 'test.txt'), 'w') as fd:
        fd.write('\n'.join(tests))


def preprocess_stage_2(path, fname_train, fname_valid, fname_test):
    def _toke_file(fname):
        toked_file = 'toked_' + fname
        bag = []
        with open(os.path.join(path, fname), 'r') as fd:
            lines = fd.read().splitlines()
            for idx, l in enumerate(lines):
                if idx % 1000 == 0:
                    print('{} toked'.format(idx))
                tokens = nltk.tokenize.word_tokenize(l)
                tokens = ' '.join(tokens)
                bag.append(tokens)

        with open(os.path.join(path, toked_file), 'w') as wfd:
            wfd.write('\n'.join(bag))

    _toke_file(fname_train)
    # _toke_file(fname_valid)
    _toke_file(fname_test)
    # substitute \ '' '' with  \"   \n with [blank]


def preprocess_stage_3_build_dict(path, dict_sz, train, valid=None, test=None):
    import operator
    d = {}

    def _count(fname, dic):
        with open(os.path.join(path, fname), 'r') as fd:
            lines = fd.read().splitlines()
            for l in lines:
                l = [w for w in l.split(' ') if w != '']
                for w in l:
                    if w in dic:
                        dic[w] += 1
                    else:
                        dic[w] = 1
        return dic

    d = _count(train, d)
    if valid is not None:
        d = _count(valid, d)
    d = _count(test, d)
    sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    sorted_d = sorted_d[:dict_sz - 3]  # leave space for sos pad unk

    keep_words = [tup[0] for tup in sorted_d]

    def _update(fname, keep_list):
        with open(os.path.join(path, fname), 'r') as fd:
            lines = fd.read().splitlines()
            new_lines = []
            for l in lines:
                l = [w for w in l.split(' ') if w != '']
                new_l = []
                for w in l:
                    if w in keep_list:
                        new_l.append(w)
                    else:
                        new_l.append('<unk>')
                new_l = ' '.join(new_l)
                new_lines.append(new_l)
        with open(os.path.join(path, 'final_' + fname), 'w') as fd:
            fd.write('\n'.join(new_lines))

    _update(train, keep_words)
    if valid is not None:
        _update(valid, keep_words)
    _update(test, keep_words)


if __name__ == "__main__":
    path = '/home/jcxu/vae_txt/data/20news'
    train_file = 'train.csv'
    test_file = 'test.csv'

    # preprocess_stage_1(path, train_file, test_file)

    fname_train = 'train.txt'
    fname_valid = 'valid.txt'
    fname_test = 'test.txt'

    # preprocess_stage_2(path,fname_train,None,fname_test)

    toked_train = 'toked_train.txt'
    toked_valid = 'toked_valid.txt'
    toked_test = 'toked_test.txt'

    # Build dictionary
    dict_size = 2000
    preprocess_stage_3_build_dict(path, dict_size, toked_train, None, toked_test)
