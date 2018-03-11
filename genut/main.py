from genut.data.load_data import *
from genut.util.argparser import ArgParser

if __name__ == "__main__":

    ap = ArgParser()

    opt = ap.parser.parse_args()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Register for logger
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    if opt.dbg is not True:
        fileHandler = logging.FileHandler("logger.log")
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    rootLogger.addHandler(consoleHandler)

    logging.info('Go!')

    if not torch.cuda.is_available():
        logging.warning('GPU NOT avail.')

    opt, data_patch = load_dataset(opt)

    logging.info(opt)

    full_embedding = load_pretrain_word_embedding(opt)

    model = Seq2seq(opt, full_embedding)
    os.chdir('../..')
    print(os.getcwd())
    if opt.use_cuda:
        model = model.cuda()
    # model = util.load_prev_model(opt.load_model, model)
    print(os.getcwd())
    if opt.load_dir is not None and opt.load_file is not None:
        model.enc = util.load_prev_state(opt.load_dir + '/' + opt.load_file + '_enc', model.enc)
        model.dec = util.load_prev_state(opt.load_dir + '/' + opt.load_file + '_dec', model.dec)
        # model.feat = util.load_prev_state(opt.load_dir + '/' + opt.load_file + '_feat', model.feat)
        try:
            model.emb = util.load_prev_state(opt.load_dir + '/' + opt.load_file + '_emb', model.emb)

        except TypeError:
            print("Trying another method")
            model.emb = util.load_prev_model(opt.load_dir + '/' + opt.load_file + '_emb', model.emb)

        try:
            model.feat = util.load_prev_model(opt.load_dir + '/' + opt.load_file + '_feat', model.feat)
        except IOError:
            print('IOError')

    print("Model Initialized.")
    if opt.mode == TEST_FLAG:
        os.chdir('/home/jcxu/ut-seq2seq/pythonrouge')
        s2s_test = summarizer.Tester(opt, model, dicts,
                                     data=test_bag, write_file='_'.join([str(opt.max_len_enc), str(opt.max_len_dec),
                                                                         str(opt.min_len_dec), opt.load_file, opt.name,
                                                                         str(opt.beam_size), str(opt.avoid), 'result']))

        s2s_test.test_iter_non_batch()

    elif opt.mode == TRAIN_FLAG:
        s2s_train = trainer.Trainer(opt, model, dicts, train_bag)
        try:
            s2s_train.train_iters()
        except KeyboardInterrupt:
            print("Training Interupted.")

    else:
        raise RuntimeError
