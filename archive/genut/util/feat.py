# Drago Feats
#  Position in Document
#  From 1 st 3 Sentences?
#  No. of Proper Nouns
#  > 20 Tokens in Sentence?
#  Sentence Length
#  No. of Co-referent Verb Mentions
#  No. of Co-referent Common Noun Mentions
#  No. of Co-referent Proper Noun Mentions
#
import numpy as np
import torch
from torch.autograd import Variable as Var


# https://www.aclweb.org/anthology/E/E14/E14-1075.pdf
# Word Locations. very significant.  Positional encoding
# Word type
# Context feature
# Unigram feature


class FeatBase():
    # this class deal with masking, mapping, scattering issue that can be used by either SparseFeat or NNFeat
    def __init__(self, opt, dicts):
        self.opt = opt
        if self.opt.feat_nn:
            self.sp = SparseFeat(opt, opt.feat_sp_dim, dicts)
            self.nn = NNFeat(opt, opt.feat_nn_dim)
        elif self.opt.feat_sp:
            self.sp = SparseFeat(opt, opt.feat_sp_dim, dicts)
            self.nn = None
        else:
            self.sp, self.nn = None, None
        self.history_max_len = None

    def compute(self, context, features, features_msk):

        if self.opt.feat_nn:
            # sparse + nn
            feat_mat = self.sp.compute(features, features_msk)
            # self.nn.compute(context, features_msk)
        elif self.opt.feat_sp:
            feat_mat = self.sp.compute(features, features_msk)
        else:
            feat_mat = None
        return feat_mat

    def update_msks(self, max_len_enc, max_len_dec, train_bag, ner_dict):
        # truncate
        if self.history_max_len == max_len_enc:
            return train_bag
        import multiprocessing
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)

        inp = [[self.opt, max_len_enc, max_len_dec, sample, ner_dict] for sample in train_bag]

        new_train_bag = pool.map(self.update_msks_batch, inp)
        pool.close()
        pool.join()

        # new_train_bag = []
        # for sample in train_bag:
        #     new_train_bag.append(util.update_msks_batch(self.opt, max_len_enc, max_len_dec, sample, ner_dict))
        return new_train_bag

    def get_block_diag(self, inp_var, ori_txt, ner_dict):
        seq_len, batch_sz = inp_var[0].size()
        seq_len = int(seq_len)
        assert batch_sz == len(ori_txt)
        # feat_mat = torch.zeros((seq_len, batch_sz, self.word_feat_num))
        whole_words, whole_sent_idxs, length_bag = util.align_original_txt(ori_txt, seq_len)

        # generate sequence for word, entity and sentence. 0, 1/2, 1/2, 0, 1, ... similar

        # Sent
        # sent_vec_msk = self.map_sent(length_bag)

        # Ent
        ent_vec_msk = util.score_msk_ent(inp_var[2], ner_dict)  # [[  [0,3],[1,5],[0,6] ,..], [7,2,1,4,]

        # Word
        word_vec_msk = util.score_msk_word(inp_var[0])  # batch, seq

        # Map them to block diagnal structure
        sent_msk = util.sent_vec2block_diag(length_bag)  # batch, seq, seq
        ent_msk = util.ent_vec2block_diag(ent_vec_msk, seq_len)  # batch, seq, seq
        # word_msk = util.vec2block_diag(word_vec_msk)
        return word_vec_msk, ent_msk, sent_msk, whole_words, length_bag

    def map_sent(self, sent_length_bag):
        """
        Given sentence split, generate a Matrix transformation. [a, b, c, d, e, f ...] => [ (a+b+c)/3, (a+b+c)/3,(a+b+c)/3 , ....]
        :param sent_length_bag:
        :return: mat: batchsz, seq
        """
        seq_len = sum(sent_length_bag[0])
        batch_sz = len(sent_length_bag)
        mat = torch.zeros(batch_sz, seq_len)
        for bidx, b in enumerate(sent_length_bag):
            tmp = torch.zeros(seq_len)
            current = 0
            for x in b:
                sent_mat = torch.ones(x) * (1. / float(x))
                tmp[current:current + x] = sent_mat
                current += x
            mat[bidx] = tmp
        return mat

    def score_msk_sent(self, sent_idxs):
        """
        Given all sentences idxs, return a msk.
        :param sent_idxs: list, batch, seq_len 000000111111122222
        :return:
        """
        batch_sz = len(sent_idxs)
        seq_len = len(sent_idxs[0])
        msk = torch.FloatTensor(batch_sz, seq_len)
        for sidx, s in enumerate(sent_idxs):
            temp = []
            temp_dcit = {}
            prev = -1
            for t in s:
                if t != prev:
                    temp_dcit[t] = 1
                else:
                    temp_dcit[t] += 1
                prev = t
            for t in s:
                temp.append(float(1. / temp_dcit[t]))

            msk[sidx] = torch.FloatTensor(np.asarray(temp))
        return msk

    def bigram_matching(self, txt, abs):
        """
        This function supports bi-gram pattern matching between txt and abs.
        eg Input: A B A C A D
            Out: A B A D
        :param txt: src_seq, batch
        :param abs: tgt_seq, batch
        :return: 1) batch_sz, seq_len, seq_len  records all the bigrams information. the first seq_len means the prev word.
                                            the second one denotes the location where there are bigrams.
                                            A[ 0 1 0 0 0 1]  note that C position is 0 since AC doesn't appear in gold summary
                                            B[ 1 0 0 0 0 0]
                                            A[ 0 1 0 0 0 1]
                                            D[ 0 0 0 0 0 0 ]
                2)  batch_sz, tgt_len           Records all
                                        [ 0 , 0 , 1 , 0 ] the first 0 is default zero.
                                        the second 0 means 'look at B, prev is A, A is a word in txt, 0 is the first time the location it appears'
                                        if the prefix word doesn't appear in the doc, it is 0. (3) will filter out this
                3)  batch_sz, tgt_len       accordingly cooperates with (2). val =1 if prefix word appear in the document
        """
        seq_len, batch_sz = txt.size()
        tgt_len, batch_sz_ = abs.size()
        assert batch_sz == batch_sz_
        meta_mat = np.zeros((batch_sz, seq_len, seq_len))
        scatter_list_for_repeat = np.zeros((batch_sz, tgt_len), dtype=int)
        tgt_msk = np.zeros((batch_sz, tgt_len), dtype=int)
        for bidx in range(batch_sz):
            doc = txt[:, bidx]
            doc_len = doc.size()[0]
            assert seq_len == doc_len
            summ = abs[:, bidx]
            tgt_seq_len = summ.size()[0]
            dict = {}

            # Record all wid -> {all occurence}

            for t in range(seq_len):
                wid = doc[t]
                if dict.has_key(wid):
                    # print(dict[wid])
                    # print(dict)
                    dict[wid] = dict[wid] + [t]
                else:
                    dict[wid] = [t]
                    # scatter_list_for_repeat[bidx][t] = dict[wid][0]
            #
            for t in range(1, tgt_seq_len):
                prev = summ[t - 1]
                if dict.has_key(prev) and prev > 30:
                    tgt_msk[bidx][t] = 1
                    scatter_list_for_repeat[bidx][t] = dict[prev][0]

            # Extract gold bigram pattern
            gold_bgrams = []
            for t in range(1, tgt_seq_len):
                if summ[t - 1] > 30:
                    gold_bgrams.append([summ[t - 1], summ[t]])
            #
            # write to matrix
            mat = np.zeros((seq_len, seq_len))
            for gold in gold_bgrams:
                prev_wid, this_wid = gold
                if dict.has_key(prev_wid) and dict.has_key(this_wid):
                    prev_all_idxs = dict[prev_wid]
                    this_all_idxs = dict[this_wid]
                    for i in prev_all_idxs:
                        for j in this_all_idxs:
                            mat[i][j] = 1
            meta_mat[bidx] = mat
        return torch.FloatTensor(meta_mat), torch.LongTensor(scatter_list_for_repeat), torch.LongTensor(tgt_msk)

    def update_msks_batch(self, opt, mode, max_len_enc, max_len_dec, current_batch, pos_dict, ner_dict):
        # Time profile result: everytime cost 0.05 secs, seems okay.

        # start = time.time()

        ori_txt = current_batch['ori_txt']
        inp_var = [current_batch['txt'], current_batch['pos'], current_batch['ner']]
        out_var = [current_batch['abs'].contiguous(), current_batch['attn']]
        # replacement = current_batch['replacement']

        cur_inp_var = util.truncate_mat(max_len_enc, inp_var)
        if mode == 0:
            cur_out_var = util.truncate_mat(max_len_dec, out_var)
            # Need to fix Attention Supervision since some of the the raw text are lost after truncation
            neg_mask = torch.ge(cur_out_var[1], max_len_enc)
            cur_out_var[1] = cur_out_var[1].masked_fill_(neg_mask, -1)
            # out_var[1] = out_var[1] * neg_mask
        else:
            cur_out_var = out_var
        real_src_len = cur_inp_var[1].size()[0]
        # Need to generate Mask for both txt and abs
        cur_inp_mask = torch.gt(cur_inp_var[0], 0)
        cur_out_mask = torch.gt(cur_out_var[0], 0)

        cur_scatter_mask = util.prepare_scatter_map(cur_inp_var[0])

        current_batch['cur_inp_var'] = cur_inp_var
        current_batch['cur_inp_mask'] = cur_inp_mask
        current_batch['cur_out_var'] = cur_out_var
        current_batch['cur_out_mask'] = cur_out_mask
        current_batch['cur_scatter_mask'] = cur_scatter_mask
        # Get stat info about bigram
        if (opt.add_loss or opt.mul_loss) and mode == 0:
            if current_batch.has_key('bigram_list'):
                meta_mat_list = current_batch['bigram_list']
                # tgt_msk = current_batch['bigram_msk']
                # dict_from_wid_to_pos_in_tgt = current_batch['bigram_dict']

                current_batch['bigram'] = util.bigramize(meta_mat_list, real_src_len, max_len_dec)
                current_batch['bigram_list'] = None
            else:
                raise NotImplementedError
                # bigram, repeat_scatter, window_msk = self.bigram_matching(cur_inp_var[0], cur_out_var[0])
                # current_batch['bigram'] = bigram
                # current_batch['repeat_scatter'] = repeat_scatter
                # current_batch['window_msk'] = window_msk

        if opt.feat_word or opt.feat_ent or opt.feat_sent:
            word_msk, ent_msk, sent_msk, words, sent_split = self.get_block_diag(cur_inp_var, ori_txt, ner_dict)
            current_batch['cur_word_msk'] = word_msk
            current_batch['cur_ent_msk'] = ent_msk
            current_batch['cur_sent_msk'] = sent_msk
            current_batch['cur_words'] = words
            current_batch['cur_sent_split'] = sent_split

        if opt.feat_sp:
            self.sp.extract_feat_batch(opt, current_batch, [pos_dict, ner_dict])
        # end = time.time()
        # print('\n')
        # print(end-start)
        return current_batch


class SparseFeat():
    def __init__(self, opt, dim, dicts):
        # FeatBase.__init__(self, opt)
        self.dim = dim
        self.opt = opt
        self.dicts = dicts[1:]
        self.word_feat_num = 17
        self.ent_feat_num = 67
        self.sent_feat_num = 12
        self.init_para(dim)

    def init_para(self, dim):
        # word 17, ent 67, sent 12
        if self.opt.feat_word:
            self.word_weight_linear = torch.nn.Linear(self.word_feat_num, dim).cuda()
            # self.word_weight_bilinear = torch.nn.Bilinear(self.word_feat_num, self.word_feat_num, dim).cuda()
        if self.opt.feat_ent:
            self.ent_weight_linear = torch.nn.Linear(self.ent_feat_num, dim).cuda()
            # self.ent_weight_bilinear = torch.nn.Bilinear(self.ent_feat_num, self.ent_feat_num, dim).cuda()
        if self.opt.feat_sent:
            self.sent_weight_linear = torch.nn.Linear(self.sent_feat_num, dim).cuda()
            # self.sent_weight_bilinear = torch.nn.Bilinear(self.sent_feat_num, self.sent_feat_num, dim).cuda()

    def ent_level_feat_tag(self, inp, dict):
        seq_len, batch_sz = inp.size()
        meta_feat = []
        for bid in range(batch_sz):
            seq = inp[:, bid]
            feat = [[] for _ in range(seq_len)]
            for t in range(seq_len):
                tag = seq[t]
                if t - 1 < 0:
                    prev_tag = 0
                else:
                    prev_tag = seq[t - 1]
                if t + 1 >= seq_len:
                    next_tag = 0
                else:
                    next_tag = seq[t + 1]
                # Uni gram
                feat[t] = feat[t] + [0 if i != tag else 1 for i in range(len(dict))]

                # bigram
                feat[t].append(0 if prev_tag == tag else 1)
                feat[t].append(0 if next_tag == tag else 1)

                # trigram
                feat[t].append(0 if prev_tag == next_tag else 1)
                feat[t].append(0 if prev_tag == tag == next_tag else 1)
            meta_feat.append(feat)
        return torch.FloatTensor(np.asarray(meta_feat)), len(meta_feat[0][0])

    def sent_level_feat(self, ner_mat, sent_split):
        seq_len, batch_sz = ner_mat.size()
        meta_feat = []

        for bidx in range(batch_sz):
            feat = []
            split = sent_split[bidx]
            ner = ner_mat[:, bidx]
            _cursor = 0
            for sidx, s in enumerate(split):
                position_in_doc = float(sidx / len(split))
                first_three = 1 if sidx < 3 else 0
                sent_len_0 = 1 if s < 4 else 0
                sent_len_1 = 1 if s < 8 else 0
                sent_len_2 = 1 if s < 16 else 0
                sent_len_3 = 1 if s < 32 else 0
                sent_len_4 = 1 if s < 64 else 0
                ner_num = torch.sum(torch.gt(ner[_cursor:_cursor + s], 0).int())
                ner_num_0 = 1 if ner_num < 1 else 0
                ner_num_1 = 1 if ner_num < 2 else 0
                ner_num_2 = 1 if ner_num < 4 else 0
                ner_num_3 = 1 if ner_num < 8 else 0
                ner_rate = float(ner_num / s)
                tmp = [position_in_doc, first_three, sent_len_0, sent_len_1, sent_len_2, sent_len_3, sent_len_4,
                       ner_num_0, ner_num_1, ner_num_2, ner_num_3, ner_rate]
                # feat = feat + tmp * s
                feat.extend([tmp for i in range(s)])
                # for k in range(s):
                #     feat[_cursor+k] = tmp
                _cursor += s
            meta_feat.append(feat)
        return torch.FloatTensor(np.asarray(meta_feat)), len(meta_feat[0][0])

    def ent_level_feat(self, inp_pos, inp_ner, dicts):
        # batch, seqlen, feat_num
        seq_len, batch_sz = inp_pos.size()
        pos_mat, pos_num = self.ent_level_feat_tag(inp_pos, dicts[0])
        ner_mat, ner_num = self.ent_level_feat_tag(inp_ner, dicts[1])
        merge_mat = torch.cat([pos_mat, ner_mat], dim=2)
        feat_num = pos_num + ner_num
        return merge_mat, feat_num

    def word_level_feat(self, inp_idx, inp_word, common_words, sent_split):

        # input is words, return feature mat: batch, seq, length_of_all_features
        seq_len, batch_sz = inp_idx.size()
        meta_feat_mat = []
        for batch_idx in range(batch_sz):
            word = inp_word[batch_idx]
            idxs = inp_idx[:, batch_idx]
            dict_of_freq = {}
            dict_of_freq_100 = {}
            feat_mat = [[] for _ in range(seq_len)]
            stop_position = min(100, seq_len)

            for t in range(seq_len):
                id = idxs[t]
                w = word[t]
                w_low = w.lower()
                # update dict
                # frequency in document
                if dict_of_freq.has_key(w_low):
                    dict_of_freq[w_low] += 1
                else:
                    dict_of_freq[w_low] = 1

                # frequency in first 100 words
                if t < stop_position:
                    if dict_of_freq_100.has_key(w_low):
                        dict_of_freq_100[w_low] += 1
                    else:
                        dict_of_freq_100[w_low] = 1

                # is common?
                feat_mat[t].append(1 if w_low in common_words else 0)
                feat_mat[t].append(1 if id < 1000 else 0)
                # is oov
                feat_mat[t].append(1 if id <= 50000 else 0)
                # word shape
                feat_mat[t] = feat_mat[t] + self.word_shape_feat(w)
                # location in document
                feat_mat[t].append(1 if float(t / seq_len) < 0.1 else 0)
                feat_mat[t].append(1 if float(t / seq_len) < 0.3 else 0)
                feat_mat[t].append(1 if t < 50 else 0)
                feat_mat[t].append(1 if t < 100 else 0)
            # frequency in document
            for t in range(seq_len):
                freq = dict_of_freq[word[t].lower()]
                feat_mat[t].append(1 if float(freq / seq_len) > 0.2 else 0)
                feat_mat[t].append(1 if float(freq / seq_len) > 0.05 else 0)
                feat_mat[t].append(1 if float(freq / seq_len) > 0.01 else 0)
            # frequency in first 100 words
            for t in range(seq_len):
                if dict_of_freq_100.has_key(word[t].lower()):

                    freq = dict_of_freq_100[word[t].lower()]
                else:
                    freq = 0.
                feat_mat[t].append(1 if float(freq / stop_position) > 0.2 else 0)
                feat_mat[t].append(1 if float(freq / stop_position) > 0.05 else 0)
                feat_mat[t].append(1 if float(freq / stop_position) > 0.01 else 0)
            _cursor = 0
            for sent_l in sent_split[batch_idx]:
                for t in range(sent_l):
                    feat_mat[t + _cursor].append(float(t / sent_l))
                _cursor += sent_l
            meta_feat_mat.append(feat_mat)
        return torch.FloatTensor(np.asarray(meta_feat_mat)), len(meta_feat_mat[0][0])

    def word_shape_feat(self, w):
        temp = [util.binary_function(w, str.isupper), util.binary_function(w, str.isalnum),
                util.binary_function(w, str.isalnum)]
        return temp

    def location_feat(self, widx, length_w, sidx, length_s):
        try:
            temp = [float(widx / length_w + 1), float(sidx / length_s + 1)]
        except ZeroDivisionError:
            return [1, 1]
        return temp

    def tag_feat(self, prev_pos, this_pos, prev_ner, this_ner):
        temp = []
        if prev_ner == this_ner:
            temp.append(0)
        else:
            temp.append(1)
        if prev_pos == this_pos:
            temp.append(0)
        else:
            temp.append(1)
        return temp

    def extract_feat_batch(self, opt, current_batch, dicts):
        # read common word list
        with open(opt.path_common,
                  'r') as rfd:
            bank_of_common_word = rfd.read().splitlines()

        cur_inp_var = current_batch['cur_inp_var']
        # cur_inp_mask = current_batch['cur_inp_mask']
        # cur_out_var = current_batch['cur_out_var']
        # cur_out_mask = current_batch['cur_out_mask']
        # cur_scatter_mask = current_batch['cur_scatter_mask']
        # word_msk = current_batch['cur_word_msk']
        # ent_msk = current_batch['cur_ent_msk']
        # sent_msk = current_batch['cur_sent_msk']
        words = current_batch['cur_words']
        sent_split = current_batch['cur_sent_split']

        if opt.feat_word:
            word_feat, num_word_feat = self.word_level_feat(cur_inp_var[0], words, bank_of_common_word, sent_split)
            current_batch['word_feat'] = word_feat
            current_batch['num_word_feat'] = num_word_feat
        else:
            current_batch['word_feat'] = None
        if opt.feat_ent:
            ent_feat, num_ent_feat = self.ent_level_feat(cur_inp_var[1], cur_inp_var[2], dicts)
            current_batch['ent_feat'] = ent_feat
            current_batch['num_ent_feat'] = num_ent_feat
        else:
            current_batch['ent_feat'] = None
        if opt.feat_sent:
            sent_feat, num_sent_feat = self.sent_level_feat(cur_inp_var[2], sent_split)
            current_batch['sent_feat'] = sent_feat
            current_batch['num_sent_feat'] = num_sent_feat
        else:
            current_batch['sent_feat'] = None
        return current_batch

    def extract_feat(self, opt, train_bag, dicts):

        # Input is a document, including words and corresponding pos tags and ner tags
        # Output is a document size matrix with indicators of each feature.
        # this function mainly composed with several sub-scoring function
        # TODO paral
        if opt.feat_nn or opt.feat_sp:
            new_bag = []
            for idx, sample in enumerate(train_bag):
                new_bag.append(self.extract_feat_batch(opt, sample, dicts))
        return new_bag

    def compute(self, features, features_msk):

        # start = time.time()
        feat_bag = []
        if self.opt.feat_word:
            word_feat = features[0] * features_msk[0].unsqueeze(2)  # batch, seq, 17
            word_feat = Var(word_feat).cuda()
            batch_sz, seq_len, word_feat_num = word_feat.size()
            word_feat = word_feat.view(batch_sz * seq_len, -1)
            w_lin = self.word_weight_linear(word_feat)
            # w_bil = self.word_weight_bilinear(word_feat, word_feat)
            # w_rep = w_lin + w_bil
            w_rep = w_lin
            feat_bag.append(w_rep)
        if self.opt.feat_ent:
            # 16, 300, 67    16, 300, 300
            ent_feature = features[1]
            batch_sz, seq_len, ent_feat_num = ent_feature.size()
            ent_feature_msk = features_msk[1]

            ent_feat = ent_feature.transpose(2, 1)
            x = torch.bmm(ent_feat, ent_feature_msk)
            x = Var(x.transpose(2, 1)).cuda()
            x = x.view(batch_sz * seq_len, -1)
            e_lin = self.ent_weight_linear(x)
            # e_bil = self.ent_weight_bilinear(x, x)
            # e_rep = e_lin + e_bil
            e_rep = e_lin
            feat_bag.append(e_rep)
        if self.opt.feat_sent:
            # feat 16 ,280, 12
            # featmsk 16, 280, 280
            sent_feature = features[2]
            sent_feature_msk = features_msk[2]
            batch_sz, seq_len, sent_feat_num = sent_feature.size()
            sent_feat = sent_feature.transpose(2, 1)
            x = torch.bmm(sent_feat, sent_feature_msk)
            x = Var(x.transpose(2, 1)).cuda()
            x = x.view(batch_sz * seq_len, -1)
            s_lin = self.sent_weight_linear(x)
            # s_bil = self.sent_weight_bilinear(x, x)
            # s_rep = s_lin + s_bil
            s_rep = s_lin
            feat_bag.append(s_rep)
        if feat_bag != []:
            rt_feat = torch.cat(feat_bag, dim=1)
        else:
            rt_feat = None
        # end = time.time()
        # print(end-start)
        return rt_feat
