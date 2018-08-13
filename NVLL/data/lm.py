import os
import torch
from NVLL.util.util import Dictionary
from NVLL.util.util import GVar


class DataLM(object):
    def __init__(self, path, batch_sz, eval_batch_sz, condition=False):
        self.condition = condition
        self.dictionary = Dictionary()
        self.test = self.tokenize(os.path.join(path, 'test.txt'), condition)
        self.train = self.tokenize(os.path.join(path, 'train.txt'), condition)
        self.dev = self.tokenize(os.path.join(path, 'valid.txt'), condition)
        print(self.dictionary.__len__())
        # self.demo_u = self.tokenize(os.path.join(path, 'test_university.txt'))
        # self.demo_u = self.set_batch(self.demo_u, 1)
        #
        # self.demo_h = self.tokenize(os.path.join(path, 'test_hotel.txt'))
        # self.demo_h = self.set_batch(self.demo_h, 1)
        #
        # self.demo_e  = self.tokenize(os.path.join(path, 'test_european.txt'))
        # self.demo_e = self.set_batch(self.demo_e, 1)

        self.dictionary.save()

        self.dev = self.set_batch(self.dev, eval_batch_sz)
        self.test = self.set_batch(self.test, eval_batch_sz)
        self.train = self.set_batch(self.train, batch_sz)  # TODO

    def tokenize(self, path,
                 condition=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        bag = []
        # Add words to the dictionary
        len_stat = []
        with open(path, 'r', errors='ignore') as f:
            for line in f:
                words = line.split()
                if len(words) < 2:
                    continue
                words = line.split() + ['<eos>']
                len_stat.append(len(words))
                # tokens += len(words)
                tmp_seq = []
                if condition:
                    tmp_seq.append(int(words[0]))
                    assert 5 > int(words[0]) >= 0
                    words = words[1:]
                for word in words:
                    self.dictionary.add_word(word)
                    tmp_seq.append(self.dictionary.word2idx[word])
                tmp_seq = torch.LongTensor(tmp_seq)
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
        if len(data_split) > 1:
            data_batch = torch.zeros((len(data_split), seq_len))
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
