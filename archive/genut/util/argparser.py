import argparse
from argparse import ArgumentParser
import os


class ArgParser():
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--start_epo', action='store', default=1, type=int)
        parser.add_argument('--avoid', action='store_true', default=False)
        parser.add_argument('--save_dir', action='store', default='/home/jcxu/vae_txt/exp-ptb', type=str)
        parser.add_argument('--mode', action='store', default=0, type=int,
                            help='TRAIN_FLAG=0, TEST_FLAG = 1')
        parser.add_argument('--dbg', action='store_true', default=False)

        parser.add_argument('--copy', action='store_true', default=False)
        parser.add_argument('--coverage', action='store_true', default=False)

        parser.add_argument('--data_path', action='store', default='/home/jcxu/vae_txt/data/ptb')

        parser.add_argument('--name', action='store', help='Name of the target domain', default='ptb')

        parser.add_argument('--src_path_vocab', action='store', default=None,
                            help='Must load with restoring model!')

        parser.add_argument('--load_dir', action='store', default=None)

        parser.add_argument('--load_file', action='store',
                            default=None,
                            help='Format like exp/12_3.45_cop... When filling, add _enc, _dec as suffix')  # 'exp_base/enc_20_3.162_50381_November24__0828')  # 'exps/enc_4_3.540_50381_November21__1059'

        parser.add_argument('--use_cuda', action='store_false')
        parser.add_argument('--feat_num', action='store', default=1, type=int,
                            help='Feature type num including original word embedding.')
        parser.add_argument('--n_epo', action='store', default=40, type=int)

        parser.add_argument('--enc', action='store', default='lstm',
                            help='lstm: Bi-LSTM; dconv: Dilated Convolution; conv')
        parser.add_argument('--dec', action='store', default='lstm')
        parser.add_argument('--att', action='store', default='general')

        parser.add_argument('--beam_size', action='store', default=5, type=int)
        parser.add_argument('--word_dict_size', action='store', default=10001, type=int)
        parser.add_argument('--ext_dict_size', action='store', type=int, default=0)
        parser.add_argument('--enc_layers', action='store', default=1, type=int)
        parser.add_argument('--dec_layers', action='store', default=1, type=int)
        parser.add_argument('--lr', action='store', default=0.01, type=float)

        parser.add_argument('--max_len_enc', action='store', default=250, type=int)
        parser.add_argument('--max_len_dec', action='store', default=60, type=int)
        parser.add_argument('--min_len_dec', action='store', default=35, type=int)

        parser.add_argument('--use_drop', action='store', default=True, type=bool)
        parser.add_argument('--use_drop_emb', action='store', type=bool, default=True)
        parser.add_argument('--dropout', action='store', type=float, default=0.5)
        parser.add_argument('--dropout_emb', action='store', type=float, default=0.5)

        parser.add_argument('--schedule', action='store', dest='schedule', type=float, default=1.,
                            help="1. means using ground truth as inpus while 0. means using predicted one.")

        parser.add_argument('--inp_dim', action='store', default=500, type=int)
        parser.add_argument('--tag_dim', action='store', default=0, type=int)
        parser.add_argument('--hid_dim', action='store', default=500, type=int)

        parser.add_argument('--clip', action='store', default=1, type=float)

        parser.add_argument('--print_every', action='store', default=100, type=int)
        parser.add_argument('--save_every', action='store', default=2000, type=int)
        parser.add_argument('--output_path', action='store', default=os.getcwd())
        parser.add_argument('--decoding', action='store', default='beam')
        parser.add_argument('--pretrain_word', action='store', default=None,
                            type=str)  # '../glove.6B.100d.txt'
        parser.add_argument('--fix_embed', action='store', default=False, type=bool)
        parser.add_argument('--path_common', action='store',
                            default='/home/jcxu/ut-seq2seq/pythonrouge/pythonrouge/RELEASE-1.5.5/data/smart_common_words.txt')

        # parser.add_argument('--dcov', action='store_true', default=False)

        parser.add_argument('--feat_nn_dim', action='store', default=None, type=int)
        parser.add_argument('--feat_nn', action='store_true', default=False)
        parser.add_argument('--feat_sp_dim', action='store', default=None, type=int)
        parser.add_argument('--feat_sp', action='store_true', default=False)

        parser.add_argument('--feat_word', action='store_true')
        parser.add_argument('--feat_ent', action='store_true')
        parser.add_argument('--feat_sent', action='store_true')
        parser.add_argument('--pe', action='store', default=False, type=bool)

        parser.add_argument('--lead3', action='store_true')

        parser.add_argument('--lw_bgdyn', default=2, type=float, action='store')
        parser.add_argument('--lw_bgsta', default=2, type=float, action='store')
        parser.add_argument('--lw_big', default=1, type=float, action='store')

        parser.add_argument('--lw_cov', action='store', default=0, type=float)
        parser.add_argument('--lw_attn', action='store', default=0.25, type=float)

        parser.add_argument('--lw_nll', action='store', default=1, type=float)

        parser.add_argument('--mul_loss', action='store_true', default=False)
        parser.add_argument('--add_loss', action='store_true', default=False)
        parser.add_argument('--big_loss', action='store_true', default=False)

        parser.add_argument('--attn_sup', action='store_true', default=False,
                            help='Add additional supervision with attention alignment.')
        parser.add_argument('--fudge', action='store', type=float, default=1e-7, help='epsilon for vae')

        parser.add_argument('--num_simu', action='store', type=int, default=4,
                            help='Time of simulation when estimating gradient for VAE.')

        self.parser = parser
