import os

os.chdir('../../data/sst')
print(os.getcwd())

idx_to_words = {}
with open("datasetSentences.txt", 'r', encoding='utf-8') as fd:
    lines = fd.read().splitlines()
    lines = lines[1:]
    for l in lines:
        idx, words = l.split("\t")
        idx_to_words[int(idx)] = words

id_to_sentiment = {}
with open("sentiment_labels.txt", 'r') as fd:
    labels = fd.read().splitlines()
    labels = labels[1:]
    for l in labels:
        id, sentiment = l.split("|")
        uniq_id = int(id)
        sentiment = float(sentiment)
        id_to_sentiment[uniq_id] = sentiment

idx_to_split = {}
with open("datasetSplit.txt", 'r') as fd:
    splits = fd.read().splitlines()
    splits = splits[1:]
    for s in splits:
        idx, section = s.split(",")
        idx_to_split[int(idx)] = int(section)

words_to_id = {}
id_to_words = {}
with open("dictionary.txt", 'r', encoding='utf-8') as fd:
    dict_entries = fd.read().splitlines()
    for d in dict_entries:
        words, uniq_id = d.split("|")
        # words = words.decode('utf8')
        uniq_id = int(uniq_id)
        words_to_id[words] = uniq_id
        id_to_words[uniq_id] = words

print("Loading finish")
print("=" * 40)

hyp_bag = []
hyp_bag.append([])
hyp_bag.append([])
hyp_bag.append([])

for idx, words in idx_to_words.items():
    sp = idx_to_split[idx]
    uniq_id = words_to_id[words]
    sentiment = id_to_sentiment[uniq_id]
    if sentiment <= 0.5:
        label = 0
    else:
        label = 1
    hyp_bag[sp - 1].append("{}\t{}".format(label, words))


def write(bag, name):
    to_wrt_string = "\n".join(bag)
    with open(name, 'w') as fd:
        fd.write(to_wrt_string)


write(hyp_bag[0], "train.txt")
write(hyp_bag[1], "test.txt")
write(hyp_bag[2], "valid.txt")
