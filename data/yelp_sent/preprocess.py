import os

label_dict = {}


def remove_label_file(filename, rm_label_fname):
    label_bag = []
    bag = []
    with open(filename, 'r') as fd:
        lines = fd.read().splitlines()
        for l in lines:
            words = l.split(" ")
            label = words[0]
            sents = words[1:]
            sent = " ".join(sents)

            if label in label_dict:
                dig = label_dict[label]
            else:
                label_dict[label] = len(label_dict)
                dig = label_dict[label]

            bag.append(sent)
            label_bag.append(str(dig))

    with open(rm_label_fname, 'w') as fd:
        fd.write("\n".join(bag))
    with open("dig_" + rm_label_fname, 'w') as fd:
        fd.write("\n".join(label_bag))


if __name__ == '__main__':
    path = os.getcwd()

    remove_label_file("label.train.txt", "train.txt")
    remove_label_file("label.valid.txt", "valid.txt")
    remove_label_file("label.test.txt", "test.txt")

    print(label_dict)
