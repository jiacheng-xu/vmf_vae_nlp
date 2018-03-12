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

        self.verbose = verbose
        self.write_file = write_file
        self.path = path
        self.dir = 'tmp' + str(random.randint(100000, 999999))
        os.mkdir(path + '/' + self.dir)
        os.mkdir(os.path.join(path, self.dir, 'ref'))
        os.mkdir(os.path.join(path, self.dir, 'sys'))

    def test_iters(self):
        try:
            # if True:
            count = 0
            test_id = 0

            # batch_order = np.arange(self.n_batch)[:10]
            batch_order = np.random.RandomState(seed=42).permutation(self.n_batch)
            # batch_order = np.random.permutation(self.n_batch)

            for idx, batch_idx in enumerate(batch_order):
                current_batch = self.test_bag[batch_idx]
                count += 1

                inp_var = current_batch['txt']
                inp_mask = current_batch['txt_msk']

                batch_size = inp_var[0].size()[1]
                assert batch_size == 1

                decoder_output, attns, p_gens = self.func_test(inp_var, inp_mask, features, feature_msks,
                                                                   max_oov_len, scatter_msk)
                decoder_output = torch.LongTensor(decoder_output).view(-1, 1)

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
