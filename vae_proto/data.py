import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.dictionary.add_word('<pad>')
        self.dictionary.add_word('<sos>')
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        bag = []
        # Add words to the dictionary
        len_stat = []
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
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
        print('Path: {}'.format(path))
        print("Number of samples: {}".format(len(bag)))
        print("Avg len: {}".format(sum(len_stat)/len(len_stat)))
        return bag
