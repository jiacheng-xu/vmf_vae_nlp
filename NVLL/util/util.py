import torch


def GVar(x):
    if torch.cuda.is_available():
        return torch.autograd.Variable(x).cuda()
    else:
        return torch.autograd.Variable(x)


def schedule(epo):
    return float(torch.sigmoid(torch.ones(1) * (epo / 3 - 5)))


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.idx_pad = 0
        self.idx_eos = 1
        self.idx_unk = 2
        self.add_word('<pad>')
        self.add_word('<eos>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
