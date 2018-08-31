import argparse


def parse_arg():
    parser = argparse.ArgumentParser(
        description='PyTorch vMF- and Gaussian- VAE LSTM language model or document model.')
    parser.add_argument('--root_path', type=str, default='/home/cc/vae_txt')
    parser.add_argument('--exp_path', type=str, default='/home/cc/exp-nvdm')
    parser.add_argument('--data_name', type=str, default='ptb', help='name of the data corpus')
    parser.add_argument('--data_path', type=str, default='data/ptb',
                        help='location of the data corpus relative to the root path.')

    parser.add_argument('--model', type=str, default='nvrnn', help='nvdm or nvrnn')

    parser.add_argument('--enc_type', type=str, default='lstm', help='lstm or gru or bow')

    parser.add_argument('--dist', type=str, default='vmf',
                        help='nor or vmf or zero or sph. '
                             'nor is gaussian; zero means not using anything from encoder side (pure lstm language model).'
                             'vmf is fixed kappa vmf; sph is dynamic kappa version of the vmf.')
    parser.add_argument('--kappa', type=float, default=0.1, help='pre-set kappa value for default vMF VAE.')

    parser.add_argument('--emsize', type=int, default=400, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=400, help='number of hidden units per layer')
    parser.add_argument('--lat_dim', type=int, default=200, help='dim of latent vec z')
    parser.add_argument('--nlayers', type=int, default=1)

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit')

    parser.add_argument('--kl_weight', type=float, default=1,
                        help='default scaling item for KL')
    parser.add_argument('--aux_weight', type=float, default=0.00001,
                        help='default scaling item for auxiliary objective term(s).  0.001 or less is good for the mu term')

    parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=3, help='evaluation batch size')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')

    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')

    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    # parser.add_argument('--exp', type=str, default='../exp', help='file dir to save and load all models and logs')
    parser.add_argument('--load', type=str, default=None, help='Name of previous model to be restored')

    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--optim', type=str, default='adam', help='sgd or adam. name of optimizer')

    parser.add_argument('--norm_func', action='store_true', default=False,
                        help='For Unif+vMF only, choose whether to use additional function to compute z_norm')

    parser.add_argument('--input_z', action='store_true', default=False,
                        help="Input the latent code z at every time step during decoding.")
    parser.add_argument('--mix_unk', type=float, default=0, help='How much of the input is mixed with UNK token')

    parser.add_argument('--swap', action='store', default=0.0, type=float,
                        help='Probability of swapping a word')
    parser.add_argument('--replace', action='store', default=0.0, type=float,
                        help='Probability of replacing a word with a random word.')

    parser.add_argument('--bi', action='store_true', default=False, help='Bidirection for encoding')

    parser.add_argument('--cd_bow', action='store', default=0, type=int, help='Condition on Bag-of-words')
    parser.add_argument('--cd_bit', action='store', default=0, type=int, help='Condition on sentiment bit')

    parser.add_argument('--board', action='store', default="board.log", type=str)
    parser.add_argument('--anneal', action='store', default=0, type=int)
    parser.add_argument('--norm_max', action='store', default=2, type=float)
    parser.add_argument('--nsample', action='store', default=3, type=int, help='Number of samples when sampling')
    # parser.add_argument('--fly', default=False, action='store_true')

    args = parser.parse_args()
    return args
