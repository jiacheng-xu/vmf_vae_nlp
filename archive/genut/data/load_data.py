import logging
import os


def load_word_dict(opt):
    """
    Load word dictionary. Naming format: str(opt.word_dict_size) + '_' + opt.name + '.vocab.dict'
    :param opt:
    :return:
    """
    fname = str(opt.word_dict_size) + '_' + opt.name + '.vocab.dict'
    logging.info('Word dict fname %s' % (fname))
    word_dict = read_bin_file(fname)
    opt.word_dict = word_dict
    opt.word_dict_size = len(word_dict)
    # opt.sos = word_dict.fword2idx('<s>')
    # opt.eos = word_dict.fword2idx('<\\s>')
    opt.unk = word_dict.fword2idx('<unk>')
    opt.pad = word_dict.fword2idx('<pad>')
    assert opt.pad == 0
    return opt


def load_tag_dict(opt, name):
    """
    Load tag dictionary.
    :param opt:
    :param name: [ ner, pos ]
    :return:
    """

    with open(name + '.dict', 'rb') as f:
        tag_dict = pkl.load(f)
    setattr(opt, name + '_dict', tag_dict)
    setattr(opt, name + '_dict_size', len(tag_dict))
    return opt


def load_data(opt):
    """
    Load dataset given mode {dbg, normal} + {train, test}
    :param opt:
    :return: data_patch
    """
    if opt.mode == 0:
        os.chdir('trains')
        train_files = os.listdir('.')
        if opt.dbg:
            files = [x for x in files if x.endswith('000.bin') or x.endswith('001.bin')]
            logging.info("DEBUG TRAIN mode: %d batch of data" % (len(files)))
        train_files = [fname for fname in train_files if fname.endswith('.bin')]
        logging.info('Total %d batch to load' % len(train_files))
        # bag = concurrent_io(read_bin_file, files)
        bag = []
        for f in train_files:
            bag.append(read_bin_file(f))

        os.chdir('..')

        os.chdir('tests')
        test_files = os.listdir('.')
        test_files = [fname for fname in test_files if fname.endswith('.bin')]
        test_bag = []
        for f in test_files:
            test_bag.append(read_bin_file(f))
        os.chdir('..')
    elif opt.mode == 1:
        os.chdir('tests')
        files = os.listdir('.')
        np.random.RandomState(seed=42).shuffle(files)
        # files = files[:200] # TODO
        if opt.dbg:
            files = files[:100]
            logging.info("DEBUG EVAL mode: %d batch of data" % (len(files)))
    else:
        logging.error('Unrecognizable mode. 0 - train 1 - test')
        raise Exception

    os.chdir('..')
    if opt.mode == 0:
        cand_ext_dict_size = reset_ext_dict_size(bag)
        if cand_ext_dict_size > opt.ext_dict_size:
            opt.ext_dict_size = cand_ext_dict_size
        logging.info("Extd dict size %s" % str(opt.ext_dict_size))
    opt.full_dict_size = opt.word_dict_size + opt.ext_dict_size
    logging.info('Full dict size: %d' % opt.full_dict_size)
    os.chdir('..')
    return opt, [bag, test_bag]


def load_pretrain_word_embedding(opt):
    """
    Loading pretrain embedding.
    :param opt:
    :return:
    """
    full_embedding = None
    if opt.dbg:
        opt.pretrain_word = None
    if opt.pretrain_word is not None:
        rng = np.random.RandomState(2018)
        full_embedding = rng.uniform(-0.3, 0.3, (opt.full_dict_size, opt.inp_dim))
        with open(opt.pretrain_word, 'r') as pretrain:
            lines = pretrain.readlines()
            for l in lines:
                x = l.split(' ')
                word = x[0]
                if opt.word_dict.has_word(word):
                    idx = opt.word_dict.fword2idx(word)
                    nums = x[1:]
                    assert len(nums) == opt.inp_dim
                    tmp_vec = []
                    for i in nums:
                        tmp_vec.append(float(i))
                    full_embedding[idx] = np.asarray(tmp_vec)
    else:
        logging.warning('Not loading pretraining word embedding.')
    return full_embedding


def load_prev_state(option, module, strict_flag=False):
    if option:
        logging.info('Loading module %s' % option)
        module.load_state_dict(torch.load(option), strict=strict_flag)
        return module
    else:
        return module


def load_dataset(opt):
    """
    Load everything needed, including dataset, dictionaries, and pre-training model.
    :param opt:
    :return:
    """
    _cur_path = os.getcwd()

    os.chdir(opt.data_path)  # chage dir
    opt = load_word_dict(opt)
    # opt = load_tag_dict(opt, 'pos')
    # opt = load_tag_dict(opt, 'ner')
    opt, data_bag = load_data(opt)
    os.chdir(_cur_path)
    return opt, data_bag
