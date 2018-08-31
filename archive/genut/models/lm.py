import torch.nn as nn
from torch.autograd import Variable

from archive.genut import SimpleRNNDecoder
from archive.genut import SingleEmbeddings


class RNNLM(nn.Module):
    def __init__(self, opt, pretrain=None):
        super(RNNLM, self).__init__()
        self.opt = opt
        self.hid_dim = opt.hid_dim
        embeds = SingleEmbeddings(opt, pretrain)
        self.emb = embeds

        rnn_dec = SimpleRNNDecoder(opt, rnn_type='lstm', input_size=opt.inp_dim,
                                   hidden_size=opt.hid_dim, emb=self.emb)

        self.dec = rnn_dec

    def forward(self, inp_var, inp_msk, tgt_var=None, tgt_msk=None, aux=None):
        batch_size = inp_var.size()[0]
        # emb = self.emb.forward(inp_var)
        # Output: Combined Word Embedding

        # Input: Combined Word Embedding  seq,batch,dim
        # h_t = self.enc.forward(emb, inp_msk)
        h_t = self.init_hidden(batch_size)
        # Output: Encoded H and h[-1]. seq,batch,dim and batch,dim

        decoder_outputs_prob, decoder_outputs = self.dec.forward(
            h_t,
            tgt_var, tgt_msk,
            aux)

        # context batch seq_len, hidden_size * num_directions )
        # hidden num_layers, seq , num_directions x hidden_size)
        return decoder_outputs_prob, decoder_outputs

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.opt.dec == 'lstm':
            return (Variable(weight.new(bsz, self.hid_dim).zero_()),
                    Variable(weight.new(bsz, self.hid_dim).zero_()))
        else:
            return Variable(weight.new(bsz, self.hid_dim).zero_())
