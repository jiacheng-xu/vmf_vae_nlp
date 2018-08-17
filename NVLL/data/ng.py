import os
import random

import numpy as np


class DataNg:
    """
    Data for 20News group or RCV1.
    Data is preprocessed by Yishu Miao.
    """

    def __init__(self, args):
        self.train = DataNg.read_data(os.path.join(args.root_path,
                                                   args.data_path, 'train.feat'))
        self.test = DataNg.read_data(os.path.join(args.root_path,
                                                  args.data_path, 'test.feat'))
        self.set_dev(1000)  # No dev set, use a part of test as dev set.
        self.test_batches = DataNg.create_batches(len(self.test[0]), args.eval_batch_size, shuffle=True)
        self.dev_batches = DataNg.create_batches(len(self.dev[0]), args.eval_batch_size, shuffle=True)
        self.read_vocab(os.path.join(args.root_path,
                                     args.data_path, 'vocab.new'))

    def read_vocab(self, path):
        with open(path, 'r') as fd:
            lines = fd.read().splitlines()
            self.vocab_size = len(lines)
            print("Vocab size: {}".format(len(lines)))

    def set_dev(self, num=100):
        l = list(range(len(self.test[0])))
        random.shuffle(l)
        l = l[:num]
        dev, dev_cnt = [], []
        for i in l:
            dev.append(self.test[0][i])
            dev_cnt.append(self.test[1][i])
        self.dev = [dev, dev_cnt]

    def set_train_batches(self, args):
        self.train_batches = DataNg.create_batches(len(self.train[0]), args.batch_size, shuffle=True)

    @staticmethod
    def read_data(path_file):
        """
        Read 20NG file
        :param path_file: Path to file
        :return: [data:a List with Dict{id:freq}, word_cnt: a List with #words in this doc]
        """
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
        return [data, word_count]

    @staticmethod
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

    @staticmethod
    def fetch_data(data, count, idx_batch, vocab_size):
        """fetch input data by batch."""
        batch_size = len(idx_batch)
        data_batch = np.zeros((batch_size, vocab_size))
        count_batch = []
        for i, doc_id in enumerate(idx_batch):
            if doc_id != -1:
                for word_id, freq in data[doc_id].items():
                    data_batch[i, word_id] = freq
                count_batch.append(count[doc_id])
            else:
                count_batch.append(0)
        return data_batch, count_batch
