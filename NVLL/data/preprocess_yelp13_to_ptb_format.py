"""
Format: [sent_bit] [w0] [w1] ...
"""


def remove_ids(fname, trunc=50):
    with open(fname, 'r', errors='ignore') as fd:
        lines = fd.read().splitlines()
    bag = []
    for l in lines:
        l = l.replace(" <sssss>", "")
        tokens = l.split("\t")
        assert len(tokens) == 7
        sent_bit = str(int(tokens[4]) - 1)
        words = tokens[6]

        txt = words.split(" ")
        if len(txt) > trunc:
            txt = txt[:trunc]
        words = " ".join(txt)
        seq = sent_bit + " " + words
        bag.append(seq)
    with open(fname[5:], 'w') as fd:
        fd.write("\n".join(bag))


import os

os.chdir("../../data/yelp")

remove_ids("yelp-test.txt")
remove_ids("yelp-train.txt")
remove_ids("yelp-valid.txt")


def check_num_words(fname):
    with open(fname, 'r') as fd:
        lines = fd.read().splitlines()
    bag = []
    for l in lines:
        words = l.split(" ")[1:]
        # words = words.split(" ")
        bag.append(len(words))
    print("{} {}".format(fname, sum(bag) / len(bag)))


check_num_words("train.txt")
check_num_words("test.txt")
check_num_words("valid.txt")


# from NVLL.util.util import Dictionary
def count(dic, fname):
    with open(fname, 'r') as fd:
        lines = fd.read().splitlines()
        lines = " ".join(lines)
        words = lines.split(" ")
        for w in words:
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1
    return dic


def reduce_vocab_sz(vocab_sz=15000):
    # pad eos unk
    d = {}
    d = count(d, "train.txt")
    d = count(d, "valid.txt")
    d = count(d, "test.txt")
    s = [(k, d[k]) for k in sorted(d, key=d.get, reverse=True)][:vocab_sz]
    rt = []
    for k, v in s:
        rt.append(k)
        # print(k, v)
    return rt


word_list = reduce_vocab_sz()


def replace(wlist, fname):
    with open(fname, 'r') as fd:
        lines = fd.read().splitlines()
        new_lines = []
        for l in lines:
            raw_words = l.split(" ")
            new_words = []
            for w in raw_words:
                if w in wlist:
                    new_words.append(w)
                else:
                    new_words.append("<unk>")
            new_lines.append(" ".join(new_words))
    with open(fname, 'w') as fd:
        fd.write("\n".join(new_lines))


replace(word_list, "train.txt")
replace(word_list, "valid.txt")
replace(word_list, "test.txt")
