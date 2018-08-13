import os


def count(dic, fname):
    with open(fname, 'r') as fd:
        lines = fd.read().splitlines()
        filtered_sents = []
        for l in lines:
            words = l.split(" ")
            _ratio = comp_unk_ratio(words)
            if _ratio <= 0.05:
                filtered_sents.append(words)
                for w in words:
                    if w in dic:
                        dic[w] += 1
                    else:
                        dic[w] = 1
    return dic, filtered_sents


def read_sent():
    pass


def comp_unk_ratio(sent):
    total = len(sent) + 0.000001
    cnt = 0
    for w in sent:
        if w == '<unk>':
            cnt += 1
    return cnt / total


def comp_ratio():
    pass


def generate_based_on_word_freq():
    count_word_freq()


def generate_based_on_sentiment():
    pass


def count_word_freq():
    d = {}
    os.chdir("../../data/yelp")
    d, _ = count(d, "valid.txt")
    d, filtered_sents_test = count(d, "test.txt")

    sorted_d = sorted(d, key=d.get, reverse=True)
    print("Len of trimmed vocab {}".format(len(sorted_d)))
    print("Num of Test samples after trimming {}".format(len(filtered_sents_test)))
    uncommon = sorted_d[-10000:]
    print(uncommon)
    divide = 5
    every = int(len(filtered_sents_test) / divide)
    sent_dictionary = {}
    for sent in filtered_sents_test:
        total = len(sent)
        cnt = 0.
        for w in sent:
            if w in uncommon:
                cnt += 1
        sent_dictionary[" ".join(sent)] = cnt / total
    sorted_sents = sorted(sent_dictionary, key=sent_dictionary.get, reverse=True)
    for piece in range(divide):
        start = int(piece * every)
        end = int((piece + 1) * every)
        tmp_sents = sorted_sents[start:end]
        with open("test-rare-" + str(piece) + ".txt", 'w') as fd:
            fd.write("\n".join(tmp_sents))


if __name__ == "__main__":
    bank_size = 1000

    # Generate 2 set of sentences.
    # Before beginning
    # if a sentence has more than 10% UNK, remove it.
    ############
    # Based on WordFreq  Vocab size=15K
    # Divide
    # Top 1K sample with largest Common Word Ratio (common word= top3K freq word)
    # Top 1K sample with largest Uncommon Word Ratio (uncommon word= top3K infreq word)
    generate_based_on_word_freq()
    ############
    # Based on Sentiment (sample from 5star and 1star)
    #############
