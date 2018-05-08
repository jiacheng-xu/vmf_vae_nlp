import logging
import os

import torch
from tensorboardX import SummaryWriter

import NVLL.argparser
from NVLL.util.gpu_flag import GPU_FLAG


def set_seed(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and GPU_FLAG:
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)


def set_save_name_log_nvdm(args):
    args.save_name = os.path.join(args.root_path, args.exp_path,
                                  'Data{}_Dist{}_Model{}_Emb{}_Hid{}_lat{}_lr{}_drop{}_kappa{}_auxw{}_normf{}'
                                  .format(
                                      args.data_name, str(args.dist), args.model,
                                      args.emsize,
                                      args.nhid, args.lat_dim, args.lr,
                                      args.dropout, args.kappa, args.aux_weight, str(args.norm_func)))
    writer = SummaryWriter(log_dir=args.save_name)
    log_name = args.save_name + '.log'
    logging.basicConfig(filename=log_name, level=logging.INFO)
    return args, writer


def set_save_name_log_nvrnn(args):
    args.save_name = os.path.join(
         args.exp_path, 'Data{}_' \
                                       'Dist{}_Model{}_Enc{}Bi{}_Emb{}_Hid{}_lat{}_lr{}_drop{}_kappa{}_auxw{}_normf{}_nlay{}_mixunk{}_inpz{}_cdbit{}_cdbow{}'
            .format(
            args.data_name, str(args.dist), args.model, args.enc_type, args.bi,
            args.emsize,
            args.nhid, args.lat_dim, args.lr,
            args.dropout, args.kappa, args.aux_weight, str(args.norm_func), args.nlayers, args.mix_unk, args.input_z,
            args.cd_bit, args.cd_bow))
    writer = SummaryWriter(log_dir=args.save_name)
    log_name = args.save_name + '.log'
    logging.basicConfig(filename=log_name, level=logging.INFO)
    return args, writer


def main():
    args = NVLL.argparser.parse_arg()
    print("GPU Flag: {}".format(GPU_FLAG))
    if args.model == 'nvdm':
        set_seed(args)
        args, writer = set_save_name_log_nvdm(args)
        print("Current dir {}".format(os.getcwd()))

        from NVLL.data.ng import DataNg
        # from NVLL.model.nvdm import BowVAE
        from NVLL.model.nvdm import BowVAE
        from NVLL.framework.run_nvdm import Runner

        data = DataNg(args)
        model = BowVAE(args, vocab_size=data.vocab_size, n_hidden=args.nhid, n_lat=args.lat_dim,
                       n_sample=5, dist=args.dist)
        if torch.cuda.is_available() and GPU_FLAG:
            print("Model in GPU")
            model = model.cuda()
        runner = Runner(args, model, data, writer)
        runner.start()
        runner.end()

    elif args.model == 'nvrnn':

        set_seed(args)
        args, writer = set_save_name_log_nvrnn(args)
        print("Current dir {}".format(os.getcwd()))

        from NVLL.data.lm import DataLM
        from NVLL.model.nvrnn import RNNVAE
        from NVLL.framework.run_nvrnn import Runner
        if args.data_name == 'ptb':
            data = DataLM(os.path.join(args.root_path, args.data_path),
                          args.batch_size,
                          args.eval_batch_size,
                          condition=False)
        elif args.data_name == 'yelp':
            data = DataLM(os.path.join(args.root_path, args.data_path),
                          args.batch_size,
                          args.eval_batch_size,
                          condition=True)
        else:
            raise NotImplementedError
        model = RNNVAE(args, args.enc_type, len(data.dictionary), args.emsize, args.nhid, args.lat_dim, args.nlayers,
                       dropout=args.dropout, tie_weights=False, input_z=args.input_z,
                       mix_unk=args.mix_unk, condition=(args.cd_bit or args.cd_bow),
                       input_cd_bow=args.cd_bow, input_cd_bit=args.cd_bit)
        # Automatic matching loading
        if args.load is not None:
            model.load_state_dict(torch.load(args.load), strict=False)
        else:
            files = os.listdir(os.path.join( args.exp_path))
            files = [f for f in files if f.endswith(".model")]
            current_name = "Data{}_Dist{}_Model{}_Enc{}Bi{}_Emb{}_Hid{}_lat{}".format(args.data_name, str(args.dist),
                                                                                      args.model, args.enc_type,
                                                                                      args.bi,
                                                                                      args.emsize,
                                                                                      args.nhid, args.lat_dim)
            for f in files:
                if current_name in f and ("mixunk0.0" in f):
                    try:
                        model.load_state_dict(torch.load(os.path.join(
                                                                      args.exp_path, f)), strict=False)
                        print("Auto Load success! {}".format(f))
                        break
                    except RuntimeError:
                        print("Automatic Load failed!")

        if torch.cuda.is_available() and GPU_FLAG:
            model = model.cuda()
        runner = Runner(args, model, data, writer)
        runner.start()
        runner.end()


if __name__ == '__main__':
    main()
