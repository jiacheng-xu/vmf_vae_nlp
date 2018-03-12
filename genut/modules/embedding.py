import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable as Var


class MultiEmbeddings(nn.Module):
    def __init__(self, opt, pretrain=None):
        super(MultiEmbeddings, self).__init__()
        self.opt = opt

        self.word_embedding = nn.Embedding(opt.full_dict_size, opt.inp_dim)
        if pretrain is not None:
            self.word_embedding.weight = nn.Parameter(torch.FloatTensor(pretrain))

        self.pos_embedding = nn.Embedding(opt.pos_dict_size, opt.tag_dim)
        self.ner_embedding = nn.Embedding(opt.ner_dict_size, opt.tag_dim)

    def forward(self, inp):
        """

        :param inp: list obj with word, pos, ner.
        :return: Concatenated word embedding. seq_len, batch_sz, all_dim
        """
        seq_word, seq_pos, seq_ner = inp
        embedded_word = self.word_embedding(seq_word)
        # print(torch.max(seq_word))
        # print(embedded_word)
        # print(torch.max(seq_pos))
        # print(torch.max(seq_ner))
        # print(self.pos_embedding)
        # print(self.ner_embedding)
        embedded_pos = self.pos_embedding(seq_pos)
        # print(embedded_pos)

        embedded_ner = self.ner_embedding(seq_ner)
        # print(embedded_ner)
        # print(torch.max(seq_ner))

        # if self.opt.dbg:
        #     final_embedding = embedded_word
        # else:
        final_embedding = torch.cat((embedded_word, embedded_pos, embedded_ner), dim=2)

        if self.opt.pe:
            seq_len, batch_sz, dim = final_embedding.size()
            position_enc = np.array(
                [[pos / np.power(10000, 2. * i / dim) for i in range(dim)] for pos in range(seq_len)])
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
            position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
            x = position_enc.unsqueeze(1)
            x = Var(x.expand_as(final_embedding)).cuda()
            final_embedding = final_embedding + 0.5 * x
        # print(final_embedding.size())
        return final_embedding

    def forward_decoding(self, inp):
        embedded_word = self.word_embedding(inp)
        return embedded_word


class SingleEmbeddings(nn.Module):
    def __init__(self, opt, pretrain=None):
        super(SingleEmbeddings, self).__init__()
        self.opt = opt
        self.drop = nn.Dropout(opt.dropout_emb)
        self.word_embedding = nn.Embedding(opt.full_dict_size, opt.inp_dim)
        if pretrain is not None:
            self.word_embedding.weight = nn.Parameter(torch.FloatTensor(pretrain))

    def forward(self, inp):
        """

        :param inp:
        :return: seq_len, batch_sz, word_dim
        """
        embedded_word = self.word_embedding(inp)
        emb = self.drop(embedded_word)
        return emb

    def forward_decoding(self, inp):
        embedded_word = self.word_embedding(inp)
        emb = self.drop(embedded_word)
        return emb
