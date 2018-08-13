# given data files, match the correct label and add it to the original file

import multiprocessing
import os


def match(data_path, key, instance):
    key_file = "label." + key + ".txt"
    # d = [[] for _ in range(5)]
    d = [[] for _ in range(50)]
    dic_label = {}
    with open(os.path.join(data_path, key_file), 'r') as fd:
        raw = fd.read().splitlines()
        for r in raw:
            words = r.split(" ")
            label = str(words[0])

            if label in dic_label:
                label_id = dic_label[label]
            else:
                dic_label[label] = len(dic_label)
                label_id = dic_label[label]
            words.append("<eos>")
            if len(words) - 1 < 10:
                s = "_".join(words[1:])
            else:
                s = "_".join(words[1:11])
            d[label_id].append(s)

    with open(instance + "logs_" + key + ".txt", 'r') as fd:
        lines = fd.read().splitlines()
    digits = []
    for l in lines:
        units = l.split("\t")
        gt = units[1].split(" ")
        if len(gt) < 10:
            s = "_".join(gt)
        else:
            s = "_".join(gt[:10])
        print(s)
        found = False
        for idx, clas in enumerate(d):
            if s in clas:
                digits.append(idx)
                d[idx].remove(s)
                found = True
                break
        if found:
            continue
        else:
            raise NotImplementedError

    with open(instance + "logs_" + key + ".lab.txt", 'w') as fd:
        assert len(lines) == len(digits)
        tmp = []
        for idx, l in enumerate(lines):
            tmp.append(str(digits[idx]) + "\t" + l)
        fd.write("\n".join(tmp))
        print("finish " + instance + "logs_" + key + ".lab.txt")


if __name__ == '__main__':
    instance_name = "/backup2/jcxu/exp-nvrnn/Datatrec_Distvmf_Modelnvrnn_EnclstmBiFalse_Emb100_Hid400_lat100_lr10.0_drop0.5_kappa50.0_auxw0.0001_normfFalse_nlay1_mixunk1.0_inpzTrue_cdbit0_cdbow0_ann0_5.093163068262161"
    data_path = "/home/jcxu/vae_txt/data/trec"

    # label.test.txt
    match(data_path, "test", instance_name)
    match(data_path, "dev", instance_name)
    match(data_path, "train", instance_name)
