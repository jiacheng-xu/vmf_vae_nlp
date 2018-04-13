import os
import torch
import logging
from NVLL.util.util import GVar
import NVLL.argparser

from tensorboardX import SummaryWriter


def set_seed(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)


def set_save_name_log(args):
    args.save_name = 'Data{}_Dist{}_Model{}_Emb{}_Hid{}_lat{}_lr{}_drop{}'.format(
        args.data_name, str(args.dist), args.model,
        args.emsize,
        args.nhid, args.lat_dim, args.lr,
        args.dropout)
    args.writer = SummaryWriter(log_dir='exps/' + args.save_name)
    log_name = args.save_name + '.log'
    logging.basicConfig(filename=log_name, level=logging.INFO)
    return args


def main():
    args = NVLL.argparser.parse_arg()
    set_seed(args)
    args = set_save_name_log(args)
    print("Current dir {}".format(os.getcwd()))
    if args.model == 'nvdm':
        from NVLL.data.ng import DataNg
        from NVLL.model.nvdm import BowVAE
        from NVLL.framework.run_nvdm import Runner

        data = DataNg(args)
        model = BowVAE(vocab_size=2000, n_hidden=args.nhid, n_lat=args.lat_dim,
                       n_sample=5, dist=args.dist)
        if torch.cuda.is_available():
            model = model.cuda()
        runner = Runner(args, model, data)
        runner.start()
        runner.end()
    elif args.model == 'nvrnn':
        from  NVLL.data.lm import DataLM
        from NVLL.model.nvrnn import RNNVAE
        from NVLL.framework.run_nvrnn import Runner
        data = DataLM(args.data_path, args.batch_size, args.eval_batch_size)
        model = RNNVAE(args, args.enc_type, len(data.dictionary), args.emsize, args.nhid, args.lat_dim, args.nlayers,
                       dropout=args.dropout, tie_weights=True)
        if torch.cuda.is_available():
            model = model.cuda()
        runner = Runner(args, model, data)
        runner.start()
        runner.end()


if __name__ == '__main__':
    main()
