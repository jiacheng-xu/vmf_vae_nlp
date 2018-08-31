class Dict(object):
    def __init__(self, bias=0):
        self.word2idx = {}
        self._idx2word = []
        self.bias = bias

    def has_word(self, word):
        if word in self.word2idx:
            return True
        else:
            return False

    def add_word(self, word):
        if word in self.word2idx:
            return self.fword2idx(word)
        else:
            l = len(self.word2idx)
            self.word2idx[word] = l
            self._idx2word.append(word)
            return self.fword2idx(word)

    def fword2idx(self, word):
        return self.word2idx[word] + self.bias

    def fidx2word(self, idx):
        return self._idx2word[idx - self.bias]

    def __len__(self):
        return len(self.word2idx)
