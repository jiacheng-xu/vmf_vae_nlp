import copy
import random

import numpy as np
import os
import torch
from torch.autograd import Variable as Var


# from pythonrouge.pythonrouge import Pythonrouge


class Tester:
    def __init__(self, opt, model, dicts, data, write_file, verbose=True, path='/backup2/jcxu/exp-cnn'):
        self.opt = opt
        self.model = model
        self.test_bag = data
        self.output_path = opt.output_path
        self.n_batch = len(data)
        self.word_dict = dicts[0]
        self.pos_dict = dicts[1]
        self.ner_dict = dicts[2]

        self.verbose = verbose
        self.write_file = write_file
        self.path = path
        self.dir = 'tmp' + str(random.randint(100000, 999999))
        os.mkdir(path + '/' + self.dir)
        # shutil.rmtree(os.path.join(path,'ref'))
        # shutil.rmtree(os.path.join(path, 'sys'))
        os.mkdir(os.path.join(path, self.dir, 'ref'))
        os.mkdir(os.path.join(path, self.dir, 'sys'))

    def test_iter_non_batch(self):
        record_str = ''
        try:
            # if True:
            count = 0
            test_id = 0
            gold_seqs = []
            pred_seqs = []

            # batch_order = np.arange(self.n_batch)[:10]
            batch_order = np.random.RandomState(seed=42).permutation(self.n_batch)
            # batch_order = np.random.permutation(self.n_batch)

            for idx, batch_idx in enumerate(batch_order):
                tmp_cur_batch = self.test_bag[batch_idx]
                count += 1
                current_batch = copy.deepcopy(tmp_cur_batch)

                current_batch = self.model.feat.update_msks_batch(
                    self.opt, self.opt.mode, self.opt.max_len_enc, self.opt.max_len_dec, current_batch, self.pos_dict,
                    self.ner_dict)

                inp_var = current_batch['cur_inp_var']
                inp_mask = current_batch['cur_inp_mask']
                scatter_msk = current_batch['cur_scatter_mask'].cuda()
                out_var = current_batch['cur_out_var']

                ori_txt = current_batch['ori_txt']

                meta = current_batch['id']

                replacement = current_batch['replacement']

                max_oov_len = len(replacement)
                batch_size = inp_var[0].size()[1]
                assert batch_size == 1
                if self.opt.feat_word or self.opt.feat_ent or self.opt.feat_sent:
                    features = [current_batch['word_feat'], current_batch['ent_feat'], current_batch['sent_feat']]
                    feature_msks = [current_batch['cur_word_msk'], current_batch['cur_ent_msk'],
                                    current_batch['cur_sent_msk']]
                else:
                    features = None
                    feature_msks = None
                if self.opt.lead3:
                    self.verbose = False
                    all_lead = ori_txt[0].split('<|||>')
                    if len(all_lead) < 3:
                        lead3 = all_lead
                    else:
                        lead3 = all_lead[:3]
                    decoder_output = inp_var[0][:40, :]
                else:
                    decoder_output, attns, p_gens = self.func_test(inp_var, inp_mask, features, feature_msks,
                                                                   max_oov_len, scatter_msk)
                    decoder_output = torch.LongTensor(decoder_output).view(-1, 1)

                #
                # decoder_output = inp_var[0][:40,:]
                #
                gold_seq, pred_seq, txt_seq = look_up_back(self.word_dict, replacement, out_var[0],
                                                           decoder_output, inp_var[0])
                if self.verbose:
                    output_string, pred_seq, gold_seq = util.demo(txt_seq, pred_seq, gold_seq, p_gens, attns,
                                                                  meta)

                    print('%d %s' % (test_id, output_string))
                    record_str += '%d %s' % (test_id, output_string)
                    # record_str += 'ROUGE: %s\n' % str(call_rouge([add_pred], [[add_gold]]))

                pred_seq[0] = util.format_seq(pred_seq[0])
                add_pred = []
                if self.opt.lead3:
                    for g in lead3:
                        add_pred.append(g.lower())
                else:
                    for s in pred_seq[0]:
                        add_pred.append(' '.join(s))

                gold_seq[0][0] = util.format_seq(gold_seq[0][0])

                add_gold = []
                for g in gold_seq[0][0]:
                    add_gold.append(' '.join(g))

                with open(os.path.join(self.path, self.dir, 'sys', str(test_id) + '_decoded.txt'), 'w') as sfd:
                    sfd.writelines('\n'.join(add_pred))
                with open(os.path.join(self.path, self.dir, 'ref', str(test_id) + '_reference.txt'), 'w') as rfd:
                    rfd.writelines('\n'.join(add_gold))
                test_id += 1

        except KeyboardInterrupt:
            print("Interrupted Keyboard!")

        result = pyrouge_eval.rouge_eval(os.path.join(self.path, self.dir, 'ref'),
                                         os.path.join(self.path, self.dir, 'sys'))
        print(result['rouge_1_f_score'], result['rouge_2_f_score'], result['rouge_l_f_score'])
        # result = call_rouge(pred_seqs, gold_seqs)

        record_str += '\n' + self.dir + '\n------------------------\n' + str(result)

        print("Saving %s" % (self.write_file))
        if self.write_file is not None:
            with open(self.write_file + '.txt', 'w') as f:
                f.write(record_str)

    def func_test(self, inp_var, inp_mask, features, feature_msks, max_oov_len, scatter_mask):
        """

        :param inp_var: (seq len, batch)
        :param inp_mask: [seq len, seq len, ...]
        :return:
        """
        # inp_var = Var(inp_var, volatile=True).cuda()
        inp_var = [Var(x, requires_grad=False).cuda() for x in inp_var]

        decoder_outputs, attns, p_gens = self.model.inference(inp_var, inp_mask, features, feature_msks, max_oov_len,
                                                              scatter_mask)

        return decoder_outputs, attns, p_gens


def loop_in_look_up_back(dict, replacement, mat, is_ref):
    # Mat: Seq_len x batch_size
    # Return bag
    max_len_replacement = len(replacement)
    word_dict_size = len(dict)
    bag_of_seq = []

    for col in range(mat.size()[1]):
        sample = []
        cur_list = []
        for row in range(mat.size()[0]):
            idx = int(mat[row, col])
            if idx == PAD:
                continue
            elif idx < word_dict_size:
                to_add = dict.fidx2word(idx)
                # cur_list.append(to_add)  #TODO
                if to_add == '<s>' or to_add == '<\s>' or to_add == '<\\s>':
                    pass
                if to_add == '.' or to_add == ';' or to_add == '!':
                    cur_list.append(to_add)
                    sample.append(cur_list)
                    cur_list = []
                else:
                    cur_list.append(to_add)
            elif idx < word_dict_size + max_len_replacement:
                cur_list.append(replacement.fidx2word(idx))
            else:
                cur_list.append('<?>')
        if cur_list == []:
            pass
        else:
            sample.append(cur_list)
        if is_ref:
            bag_of_seq.append([sample])
        else:
            bag_of_seq.append(sample)
    return bag_of_seq


def look_up_back(dict, replacement, gold, pred, txt=None):
    batch_size = pred.size()[1]
    batch_size_ = gold.size()[1]
    max_seq_len = gold.size()[0]
    assert batch_size == batch_size_
    gold_seq = loop_in_look_up_back(dict, replacement, gold, True)
    pred_seq = loop_in_look_up_back(dict, replacement, pred, False)
    txt_seq = None
    if txt is not None:
        txt_seq = loop_in_look_up_back(dict, replacement, txt, False)
    return gold_seq, pred_seq, txt_seq


def call_rouge(summary, reference, bool_length_limit=False, length_limit=10):
    # ROUGE_dir = './pythonrouge/pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl'
    # data_dir = './pythonrouge/pythonrouge/RELEASE-1.5.5/data/'
    # print('Calling ROUGE')
    rouge = pythonrouge.Pythonrouge(summary_file_exist=False,
                                    summary=summary, reference=reference,
                                    n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                                    recall_only=False, stemming=True, stopwords=False, use_cf=True, ROUGE_W=False,
                                    ROUGE_W_Weight=1.3, word_level=True, length_limit=bool_length_limit,
                                    length=length_limit)

    score = rouge.calc_score()
    return score


def test_rouge():
    ROUGE_dir = './pythonrouge/pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl'
    data_dir = './pythonrouge/pythonrouge/RELEASE-1.5.5/data/'
    import os
    # system summary & reference summary
    summary = [[
        " button denied 100th race start for mclaren after ers failure .",
        " button then spent much of the bahrain grand prix on twitter delivering his verdict on the action as it unfolded .",
        " lewis hamilton has out-qualified and finished ahead of mercedes team-mate nico rosberg at every race this season .",
        " bernie ecclestone confirms f1 will make its bow in azerbaijan next season ."]]
    # summary = [["better boy","cat ","better option"]]
    reference = [[[
        "button was denied his 100th race for mclaren .",
        " ers prevented him from making it to the start-line .",
        " the briton.",
        " he quizzed after nico rosberg accused lewis hamilton of pulling off such a manoeuvre in china .",
        " button has been in azerbaijan for the first time since 2013 ."]]]
    # initialize setting of ROUGE to eval ROUGE-1, 2, SU4
    # if you evaluate ROUGE by sentence list as above, set summary_file_exist=False
    # if recall_only=True, you can get recall scores of ROUGE
    rouge = pythonrouge.Pythonrouge(summary_file_exist=False,
                                    summary=summary, reference=reference,
                                    n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                                    recall_only=True, stemming=True, stopwords=True,
                                    word_level=True, length_limit=False, length=50,
                                    use_cf=False, cf=95, scoring_formula='average',
                                    resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    print(score)


def sent_modifier(sent):
    sent = sent.strip()
    ss = sent.split(' ')
    new_ss = []
    prev_bool = False
    eos_pos = []
    for idx, word in enumerate(ss):
        if word == '<\s>':  # TODO
            eos_pos.append(idx)
        if word == ',' or word == '.':
            if prev_bool == False:
                prev_bool = True
                new_ss.append(word)
            else:
                continue
        else:
            prev_bool = False
            new_ss.append(word)
    if len(eos_pos) > 4:
        new_ss = new_ss[:eos_pos[3]]
    rt = ''
    for i in new_ss:
        rt += i + ' '
    return rt.strip()


def test_AS(path):
    import os
    path_art = 'articles'
    path_base = 'baseline'
    path_point = 'pointer-gen'
    path_cov = 'pointer-gen-cov'
    path_ref = 'reference'
    path_sys = path_base
    # 000000_decoded.txt
    # 000000_reference.txt
    os.chdir(path)
    syss = [path_sys + '/' + x for x in os.listdir(path_sys)]
    summary = [[] for _ in range(len(syss))]
    refs = [x for x in os.listdir(path_ref)]
    reference = [[[]] for _ in range(len(refs))]
    assert len(syss) == len(refs)
    for r in refs:
        fid = r.split('_')[0]
        with open(path_ref + '/' + r, 'r') as rfd:
            lines = rfd.read().splitlines()
            for l in lines:
                reference[int(fid)][0].append(l)
        with open(path_sys + '/' + str(fid) + '_decoded.txt', 'r') as sfd:
            ls = sfd.read().splitlines()
            for l in ls:
                summary[int(fid)].append(l)

    rouge = pythonrouge.Pythonrouge(summary_file_exist=False,
                                    summary=summary, reference=reference,
                                    n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                                    recall_only=True, stemming=True, stopwords=True,
                                    word_level=True, length_limit=False, length=50,
                                    use_cf=False, cf=95, scoring_formula='average',
                                    resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    print(score)

# test_AS('/home/jcxu/test_output')

# test_rouge()
