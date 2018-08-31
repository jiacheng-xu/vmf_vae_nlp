import torch.nn as nn
from archive.genut import SingleEmbeddings
from archive.genut.modules.enc import RNNEncoder
from archive.genut.modules.dec.decoder import RNNDecoder


class Seq2seq(nn.Module):
    def __init__(self, opt, pretrain=None):
        super(Seq2seq, self).__init__()
        self.opt = opt

        embeds = SingleEmbeddings(opt, pretrain)
        self.emb = embeds

        if self.opt.enc == 'lstm':
            enc = RNNEncoder(opt, opt.inp_dim + opt.tag_dim * 2, opt.hid_dim, rnn_type=opt.enc.lower())
        elif self.opt.enc == 'dconv':
            raise NotImplementedError
            # enc = Encoder.DCNNEncoder()
        elif self.opt.enc == 'conv':
            raise NotImplementedError
            # enc = Encoder.CNNEncoder(inp_dim=opt.inp_dim + opt.tag_dim * 2, hid_dim=opt.hid_dim, kernel_sz=5, pad=2,
            #                          dilat=1)
        else:
            raise NotImplementedError
        self.enc = enc

        # self.feat = Feat.FeatBase(opt, opt.dicts)
        self.feat = None  # TODO feature unavail

        rnn_dec = RNNDecoder(opt, rnn_type='lstm', num_layers=1,
                             hidden_size=opt.hid_dim, input_size=opt.inp_dim, attn_type='general',
                             coverage=opt.coverage,
                             copy=opt.copy, dropout=opt.dropout, emb=self.emb,
                             full_dict_size=self.opt.full_dict_size,
                             word_dict_size=self.opt.word_dict_size,
                             max_len_dec=opt.max_len_dec)

        self.dec = rnn_dec

    def forward(self, inp_var, inp_msk, tgt_var=None, tgt_msk=None, aux=None):
        if self.training:  # train mode
            self.forward_train(inp_var, inp_msk, tgt_var, tgt_msk, aux)
        else:  # eval mode
            self.forward_eval(inp_var, inp_msk, aux)

    def forward_train(self, inp_var, inp_msk, tgt_var, tgt_msk, aux):

        # def forward_train(self, inp_var, tgt_var, inp_mask, tgt_mask, features, feature_msks, max_oov_len, scatter_mask,
        #                   bigram_bunch,logger):
        # Input: WordIdx, PosIdx, NerIdx
        emb = self.emb.forward(inp_var)
        # Output: Combined Word Embedding

        # Input: Combined Word Embedding  seq,batch,dim
        context, h_t = self.enc.forward(emb, inp_msk)
        # Output: Encoded H and h[-1]. seq,batch,dim and batch,dim

        # Run sparse feature encoder

        # if self.feat != None:     # TODO feature module currently unavail
        #     feats = self.feat.compute(context, features, feature_msks)
        # else:
        #     feats = None

        decoder_outputs_prob, decoder_outputs, attns, discount, loss_cov, p_copys = self.dec.forward(context, inp_msk,
                                                                                                     h_t,
                                                                                                     tgt_var, tgt_msk,
                                                                                                     inp_var, aux)

        # context batch seq_len, hidden_size * num_directions )
        # hidden num_layers, seq , num_directions x hidden_size)
        return decoder_outputs_prob, decoder_outputs, attns, discount, loss_cov, p_copys

    def forward_eval(self, inp_var, inp_mask, aux):
        """

        :param inp_var: (seq len, batch size)
        :param inp_mask: [seq len, ....]
        :return:
        """
        emb = self.emb.forward(inp_var)

        context, h_t = self.enc.forward(emb, inp_mask)
        # output: PackedSequence(seq_len * batch, hidden_size * num_directions),
        # hidden: tupple((batch, hidden_size * 2), ....)
        if self.feat != None:
            feats = self.feat.compute(context, features, feature_msks)
        else:
            feats = None
        # context, context_mask_ = nn.utils.rnn.pad_packed_sequence(context)
        contxt_len, batch_size, hdim = context.size()
        context_len__, batch_size__ = inp_var[0].size()
        assert batch_size == 1
        assert context_len__ == contxt_len
        decoder_outputs, attns, p_gens = self.dec.beam_decode(context, inp_mask, h_t, inp_var[0], feats,
                                                              max_oov_len,
                                                              scatter_mask)
        # decoder_outputs_prob, _, decoder_outputs, attn_bag = self.dec.greedy_decode(context, inp_mask, hidden,
        #                                                                             max_len=120)
        return decoder_outputs, attns, p_gens
