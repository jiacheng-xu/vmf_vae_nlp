import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

import unittest


class CNNEncoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, kernel_sz, pad, dilat):
        super(CNNEncoder, self).__init__()
        self.encoder = torch.nn.Conv1d(in_channels=inp_dim, out_channels=hid_dim, kernel_size=kernel_sz, stride=1,
                                       padding=pad, dilation=dilat)

    def forward(self, inp, inp_mask):
        # seq,batch,dim
        inp = inp.permute(1, 2, 0)
        # batch, dim, seq
        # print('1')
        x = torch.nn.functional.relu(self.encoder(inp))
        # batch, dim, seq
        # seq,batch,dim
        # print('1')
        x = x.permute(2, 0, 1)
        h_t = (x[-1], x[-1])
        # print('1')
        return x, h_t
        # print(h_t.size())
        # print(x)


class DCNNEncoder(nn.Module):
    def __init__(self, inp_dim, hid_dim=150, kernel_sz=5, pad=2, dilat=1):
        super(DCNNEncoder, self).__init__()
        self.encoder = torch.nn.Conv1d(in_channels=inp_dim, out_channels=hid_dim, kernel_size=kernel_sz, stride=1,
                                       padding=pad, dilation=1)

    def forward(self, inp, mask):
        inp = inp.permute(1, 2, 0)
        x = torch.nn.functional.relu(self.encoder(inp))
        print(x.size())


class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == "__main__":
    unittest.main()
    batch = 16
    seq = 200
    dim = 400

    inp = torch.autograd.Variable(torch.rand((seq, batch, dim)))
    cnn = DCNNEncoder(inp_dim=dim, hid_dim=150, kernel_sz=5, pad=2, dilat=1)
    cnn.forward(inp, 0)
