import os
import torch
from NVLL.util.util import Dictionary
from NVLL.util.util import GVar


class DataLM(object):
    def __init__(self, path, batch_sz, eval_batch_sz):
        self.dictionary = Dictionary()
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.train = self.tokenize(os.path.join(path, 'train.txt'))  # TODO
        self.dev = self.tokenize(os.path.join(path, 'valid.txt'))

        self.dev = self.set_batch(self.dev, eval_batch_sz)
        self.test = self.set_batch(self.test, eval_batch_sz)
        # self.train = self.dev
        self.train = self.set_batch(self.train, batch_sz)  # TODO

    def tokenize(self, path):

        """Tokenizes a text file."""
        assert os.path.exists(path)
        bag = []
        # Add words to the dictionary
        len_stat = []
        with open(path, 'r') as f:
            for line in f:
                words = line.split()
                if len(words) <= 1:
                    continue
                words = line.split() + ['<eos>']
                len_stat.append(len(words))
                # tokens += len(words)
                tmp_seq = []
                for word in words:
                    self.dictionary.add_word(word)
                    tmp_seq.append(self.dictionary.word2idx[word])
                tmp_seq = torch.LongTensor(tmp_seq)
                # label = None
                # bag.append([tmp_seq, label])
                bag.append(tmp_seq)

        bag = sorted(bag, key=lambda sample: sample.size()[0], reverse=True)
        print('Path: {} Min length: {}'.format(path, bag[-1].size()[0]))
        print("Number of samples: {}".format(len(bag)))
        print("Avg len: {}".format(sum(len_stat) / len(len_stat)))
        return bag

    def set_batch(self, data_bag, batch_sz, batch_first=False):
        rt_bag_of_tensor = []

        total_len = len(data_bag)
        assert batch_first == False
        cnt = 0
        while (cnt + 1) * batch_sz < total_len:
            data_split = data_bag[cnt * batch_sz: (cnt + 1) * batch_sz]
            seq_len = data_split[0].size()[0]

            data_batch = torch.zeros((batch_sz, seq_len))
            for i, doc in enumerate(data_split):
                for t in range(doc.size()[0]):
                    data_batch[i][t] = doc[t]
            data_batch = torch.transpose(data_batch, 1, 0).long()
            rt_bag_of_tensor.append(data_batch)
            cnt += 1
        data_split = data_bag[cnt * batch_sz:]
        seq_len = data_split[0].size()[0]

        data_batch = torch.zeros((batch_sz, seq_len))
        for i, doc in enumerate(data_split):
            for t in range(doc.size()[0]):
                data_batch[i][t] = doc[t]
        data_batch = torch.transpose(data_batch, 1, 0).long()
        rt_bag_of_tensor.append(data_batch)

        return rt_bag_of_tensor

    @staticmethod
    def get_feed(data_patch):
        # seq, batch
        bsz = data_patch.size()[1]
        sos = torch.LongTensor(1, bsz).fill_(1)
        input_data = GVar(torch.cat((sos, data_patch[:-1])))
        return input_data
