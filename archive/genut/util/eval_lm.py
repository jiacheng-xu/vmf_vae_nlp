from torch import nn
import numpy as np
from torch.autograd import Variable as Var
import math
import logging
# from pythonrouge.pythonrouge import Pythonrouge
from archive.genut import msk_list_to_mat


class Tester:
    def __init__(self, opt, model, data, write_file, verbose=True, path='/home/jcxu/exp-ptb'):
        self.opt = opt
        self.model = model
        self.test_bag = data
        self.output_path = opt.output_path
        self.n_batch = len(data)
        self.word_dict = opt.word_dict

        self.verbose = verbose
        self.write_file = write_file
        self.path = path
        self.crit = nn.CrossEntropyLoss(size_average=True, ignore_index=0)

    def test_iters(self):
        try:
            # if True:
            count = 0
            test_id = 0

            batch_order = np.random.RandomState(seed=42).permutation(self.n_batch)
            # batch_order = np.random.permutation(self.n_batch)

            accumulated_ppl = 0
            test_len = 0
            for idx, batch_idx in enumerate(batch_order):
                current_batch = self.test_bag[batch_idx]
                count += 1

                inp_var = current_batch['txt']
                inp_mask = current_batch['txt_msk']
                test_len += inp_mask[0]
                batch_size = inp_var.size()[0]
                assert batch_size == 1

                nll, decoder_output = self.func_test(inp_var, inp_mask)
                accumulated_ppl += nll
                logging.info(nll)

                test_id += 1

        except KeyboardInterrupt:
            print("Interrupted Keyboard!")

        final_ppl = accumulated_ppl / test_len
        return math.exp(final_ppl)

    def func_test(self, inp_var, inp_msk):
        target_len = inp_msk[0]
        batch_size = inp_var.size()[0]

        decoder_outputs_prob, decoder_outputs = self.model.forward(inp_var, inp_msk, tgt_var=inp_var, tgt_msk=inp_msk,
                                                                   aux=None)

        valid_pos_mask = Var(msk_list_to_mat(inp_msk), requires_grad=False).view(target_len * batch_size, 1)
        if self.opt.use_cuda:
            valid_pos_mask = valid_pos_mask.cuda()

        # Compulsory NLL loss part
        pred_prob = decoder_outputs_prob.view(target_len * batch_size, -1)
        seq_first_inp_var = inp_var.transpose(1, 0).contiguous()
        gold_dist = Var(seq_first_inp_var.view(target_len * batch_size))
        if self.opt.use_cuda:
            gold_dist = gold_dist.cuda()

        loss = self.crit(pred_prob, gold_dist)
        loss = loss * target_len
        return loss.data[0], decoder_outputs
