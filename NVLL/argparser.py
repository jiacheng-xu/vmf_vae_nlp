import argparse


def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch VAE LSTM Language Model')

    parser.add_argument('--data_name', type=str, default='ptb', help='name of the data corpus')
    parser.add_argument('--data_path', type=str, default='../data/ptb', help='location of the data corpus')

    parser.add_argument('--fly', default=False, action='store_true')
    parser.add_argument('--enc_type', type=str, default='lstm', help='lstm or bow')
    parser.add_argument('--model', type=str, default='nvrnn', help='nvdm or nvrnn')
    parser.add_argument('--distribution', type=str, default='vmf', help='nor or vmf or zero')
    parser.add_argument('--kappa', type=float, default=5)

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

    parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=3, help='evaluation batch size')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')

    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')

    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    # parser.add_argument('--log-interval', type=int, default=100, metavar='N',
    #                     help='report interval')

    parser.add_argument('--exp', type=str, default='../exp', help='file dir to save and load all models and logs')
    parser.add_argument('--load', type=str, default=None, help='Name of previous model to be restored')

    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--optim', type=str, default='adam', help='sgd or adam')

    parser.add_argument('--klw_bound', type=float, default=1., help='Upper bound for weight for kl term.')
    parser.add_argument('--mean_reg', type=float, default=0.001)

    args = parser.parse_args()
    return args
