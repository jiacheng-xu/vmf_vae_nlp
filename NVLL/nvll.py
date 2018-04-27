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


def set_save_name_log_nvdm(args):
    args.save_name = '/backup2/jcxu/exp-nvdm/Data{}_Dist{}_Model{}_Emb{}_Hid{}_lat{}_lr{}_drop{}_kappa{}_auxw{}_normf{}'.format(
        args.data_name, str(args.dist), args.model,
        args.emsize,
        args.nhid, args.lat_dim, args.lr,
        args.dropout, args.kappa,args.aux_weight,str(args.norm_func))
    writer = SummaryWriter(log_dir=args.save_name)
    log_name = args.save_name + '.log'
    logging.basicConfig(filename=log_name, level=logging.INFO)
    return args, writer


def set_save_name_log_nvrnn(args):
    args.save_name = '/backup2/jcxu/exp-nvrnn/Data{}_' \
                     'Dist{}_Model{}_Emb{}_Hid{}_lat{}_lr{}_drop{}_kappa{}_auxw{}_normf{}_nlay{}_mixunk{}_inpz{}'.format(
        args.data_name, str(args.dist), args.model,
        args.emsize,
        args.nhid, args.lat_dim, args.lr,
        args.dropout, args.kappa,args.aux_weight,str(args.norm_func),args.nlayers, args.mix_unk, args.input_z)
    writer = SummaryWriter(log_dir=args.save_name)
    log_name = args.save_name + '.log'
    logging.basicConfig(filename=log_name, level=logging.INFO)
    return args, writer


def main():
    args = NVLL.argparser.parse_arg()
    if args.model == 'nvdm':
        set_seed(args)
        args, writer = set_save_name_log_nvdm(args)
        print("Current dir {}".format(os.getcwd()))

        from NVLL.data.ng import DataNg
        # from NVLL.model.nvdm import BowVAE
        from NVLL.model.nvdm_v2 import BowVAE
        from NVLL.framework.run_nvdm import Runner

        data = DataNg(args)
        model = BowVAE(args, vocab_size=data.vocab_size, n_hidden=args.nhid, n_lat=args.lat_dim,
                       n_sample=5, dist=args.dist)
        if torch.cuda.is_available():
            model = model.cuda()
        runner = Runner(args, model, data, writer)
        runner.start()
        runner.end()

    elif args.model == 'nvrnn':

        set_seed(args)
        args, writer = set_save_name_log_nvrnn(args)
        print("Current dir {}".format(os.getcwd()))


        from  NVLL.data.lm import DataLM
        from NVLL.model.nvrnn import RNNVAE
        from NVLL.framework.run_nvrnn import Runner
        data = DataLM(args.data_path, args.batch_size, args.eval_batch_size)
        model = RNNVAE(args, args.enc_type, len(data.dictionary), args.emsize, args.nhid, args.lat_dim, args.nlayers,
                       dropout=args.dropout, tie_weights=False,input_z=args.input_z, mix_unk=args.mix_unk)
        if args.load is not None:
            model.load_state_dict(torch.load(args.load),strict=False)
        if torch.cuda.is_available():
            model = model.cuda()
        runner = Runner(args, model, data, writer)
        runner.start()
        runner.end()


if __name__ == '__main__':
    main()
