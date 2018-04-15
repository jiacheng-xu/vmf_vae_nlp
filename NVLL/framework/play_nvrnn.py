# Load nvdm or nvrnn, check how Gauss or vMF distributes
"Dataptb_Distnor_Modelnvrnn_Emb400_Hid400_lat200_lr0.001_drop0.2"

import logging
import math
import random
import time
import numpy
import torch
from  NVLL.data.lm import DataLM
from NVLL.model.nvrnn import RNNVAE
from NVLL.framework.run_nvrnn import Runner
from NVLL.util.util import schedule, GVar
from NVLL.model.nvrnn import RNNVAE


class PlayNVRNN():
    def __init__(self):
        self.args = self.load_args()
        self.model = self.load_model()
        self.data = self.load_data()

    def load_data(self):
        data = DataLM(self.args.data_path, self.args.batch_size, self.args.eval_batch_size)
        return data

    def load_args(self):
        pass

    def load_model(self):
        pass

    def evaluate(self):
        # Load the best saved model.
        model = RNNVAE(self.args, self.args.enc_type, len(self.data.dictionary), self.args.emsize,
                       self.args.nhid, self.args.lat_dim, self.args.nlayers,
                       dropout=self.args.dropout, tie_weights=True)
        model.load_state_dict(torch.load(self.args.save_name))
        model = model.cuda()
        print(model)
        print(self.args)
        # with open(self.args.save_name, 'rb') as f:
        #     model = torch.load(f)
        cur_loss, cur_kl, test_loss = self.evaluate(self.args, model,
                                                    self.data.test)
        Runner.log_eval(cur_loss, cur_kl, test_loss, True)
        self.writer.close()
