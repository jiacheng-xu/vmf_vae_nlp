import argparse


def parse_arg():
    parser = argparse.ArgumentParser(description='Analyzer')
    parser.add_argument('--board', type=str, default='ana_board.log')
    parser.add_argument('--root_path', type=str, default='/home/cc/vae_txt')
    parser.add_argument('--exp_path', type=str, default='/home/cc/exp-nvrnn')
    parser.add_argument('--instance_name', type=str)
    parser.add_argument('--data_path', type=str, default='data/ptb', help='location of the data corpus')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='evaluation batch size')

    parser.add_argument('--mix_unk', type=float, default=0)
    parser.add_argument('--swap', action='store', default=0.2, type=float,
                        help='Probability of swapping a word')
    parser.add_argument('--replace', action='store', default=0, type=float,
                        help='Probability of replacing a word with a random word.')

    parser.add_argument('--cd_bow', action='store', default=0, type=int)
    parser.add_argument('--cd_bit', action='store', default=0, type=int)
    parser.add_argument('--temp', action='store', default=1, type=float)

    parser.add_argument('--split', action='store', default=0, type=int)

    args = parser.parse_args()
    return args
