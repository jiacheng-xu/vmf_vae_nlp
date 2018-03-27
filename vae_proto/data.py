import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.idx_pad = 0
        self.idx_sos = 1
        self.idx_unk = 2
        self.add_word('<pad>')
        self.add_word('<sos>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, start_idx=0,end_idx=0):
        self.dictionary = Dictionary()
        self.test = self.tokenize(os.path.join(path, 'test.txt'), start_idx,end_idx)
        self.train = self.tokenize(os.path.join(path, 'train.txt'), start_idx,end_idx)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), start_idx,end_idx)


    def tokenize(self, path, start_idx, end_idx):

        """Tokenizes a text file."""
        assert os.path.exists(path)
        bag = []
        # Add words to the dictionary
        len_stat = []
        with open(path, 'r') as f:
            for line in f:
                words = line.split()[start_idx:]
                if end_idx != 0 and len(words)> end_idx:
                    words = words[:end_idx]
                if len(words) <= 1:
                    continue
                # words = line.split() + ['<eos>']
                len_stat.append(len(words))
                # tokens += len(words)
                tmp_seq = []
                for word in words:
                    self.dictionary.add_word(word)
                    tmp_seq.append(self.dictionary.word2idx[word])
                tmp_seq = torch.LongTensor(tmp_seq)
                label = None
                bag.append([tmp_seq, label])

        bag = sorted(bag, key=lambda sample: sample[0].size()[0],reverse=True)
        print('Min length: {}'.format(bag[-1][0].size()[0]))
        print('Path: {}'.format(path))
        print("Number of samples: {}".format(len(bag)))
        print("Avg len: {}".format(sum(len_stat)/len(len_stat)))
        return bag
