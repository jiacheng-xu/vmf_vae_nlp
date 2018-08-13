import abc
import multiprocessing
import os
import random
import numpy as np
import scipy
import scipy.spatial
from numpy import linalg as LA


def comp_jaccard_distance(a, b):
    """

    :param a: a list of words
    :param b: a list of words
    :return: Jaccard distance between these two lists
    """
    raw_len = len(a) + len(b)
    cnt = 0
    for w in a:
        if w in b and w not in ["a", "the", "is", "was", "they", "i", "an", "all", "he", "she", "<unk>", "<eos>"]:
            cnt += 1.
    return (cnt) / (raw_len - cnt)


class DistAnalyzer():
    @staticmethod
    def line_to_numpy(line):
        nums = line.split("\t")
        num_list = [float(n) for n in nums]
        return np.asarray(num_list)

    @staticmethod
    def read_sample(fname):
        pass

    def batch_read_sample(self, path):
        log_names = [os.path.join(path, i) for i in os.listdir(path) if i.endswith(".txt")]
        # self.read_sample(log_names[0])
        cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=cores) as pool:
            result = pool.map(self.read_sample, log_names)
        return result


class GaussAnalyzer(DistAnalyzer):
    def __init__(self, path, sample_num=50):
        self.path = path
        self.data = self.batch_read_sample(path)
        self.N = sample_num
        print("Gauss: Finish loading data. Start Analyzing!")
        # self.analyze()
        self.distance_compare()

    @staticmethod
    def distance_compare_unit(data):
        host = data[0]
        host_words = host['gt'].split(" ")
        host_mean = host['mean']
        host_mean = host_mean / LA.norm(host_mean)

        guest = data[1:]
        num_guest = len(guest)
        bag = []
        for idx, g in enumerate(guest):
            g_mean = g['mean']
            g_words = g['gt'].split(" ")
            # jaccard = comp_jaccard_distance(g_words, host_words)
            cos_distance = -1 * (scipy.spatial.distance.cosine(g_mean, host_mean) - 1)
            bag.append([0, cos_distance])
        return bag

    def distance_compare(self, num=50):
        print("Start distance comparison. ")
        all_data_size = len(self.data)
        results = []
        for n in range(num):
            x = random.sample(range(all_data_size), num)
            this_data = [self.data[i] for i in x]
            point = self.distance_compare_unit(this_data)
            results += point
        results = sorted(results, key=lambda x: x[0], reverse=True)

        for r in results:
            # print("{}\t{}".format(r[0], r[1]))
            # print(r[1])
            print((r[1] + 1) // 0.1)
        # print("First num: Jaccard Distance; Second num: Cos distance (smaller closer) (range:0-2)")

    def analyze(self):
        # For Gauss, sample 20 sents close to origin and 20 far from origin.
        # Sort according to norm, show the norm of [0-25%),[25%-50%),[50%-75%),[75%-100%) and their logvar
        # Return their uniq-id
        sorted_data = sorted(self.data, key=lambda sample: sample['norm_mean'], reverse=True)
        num_of_samples = len(sorted_data)
        divide = 10
        piece = num_of_samples // divide
        bag = []
        for idx in range(divide):
            bag.append(sorted_data[piece * idx:(idx + 1) * piece])

        ###################
        # Closest and farthest
        closest = sorted_data[-self.N:]
        farthest = sorted_data[:self.N]
        print("------------------------------Print farthest samples------------------------------")
        for x in farthest:
            print(self.print_sample(x))
        print("------------------------------END   farthest samples------------------------------")
        print("-" * 100)
        print("------------------------------Print closest samples------------------------------")
        for x in closest:
            print(self.print_sample(x))
        print("------------------------------END   closest samples------------------------------")
        ###################
        print("------------------------------Relation between Norm of Mean and Others------------------------------")
        for b in bag:
            avg_mean, avg_logvar, avg_len, avg_recon, avg_kl, avg_nll = self.relation_of_norm_mean_and_x(b)
            print("avg_mean: {}\tavg_logvar: {}\tavg_len: {}\tavg_recon: {}\tavg_kl: {}\tavg_nll: {}".
                  format(avg_mean, avg_logvar, avg_len, avg_recon, avg_kl, avg_nll))

    def relation_of_norm_mean_and_x(self, batch):
        cnt = len(batch)
        sum_mean = 0
        sum_logvar = 0
        sum_len = 0
        sum_recon = 0
        sum_kl = 0
        sum_nll = 0
        for sample in batch:
            sum_mean += sample['norm_mean']
            sum_logvar += sample['norm_logvar']
            sum_len += len(sample['gt'].split(" "))
            sum_recon += sample['recon']
            sum_kl += sample['kl']
            sum_nll += sample['nll']
        avg_mean = sum_mean / cnt
        avg_logvar = sum_logvar / cnt
        avg_len = sum_len / cnt
        avg_recon = sum_recon / cnt
        avg_kl = sum_kl / cnt
        avg_nll = sum_nll / cnt
        return avg_mean, avg_logvar, avg_len, avg_recon, avg_kl, avg_nll

    @staticmethod
    def print_sample(sample):
        s = "===============\n"
        s += "ID: {}\n".format(sample['id'])
        s += "GT:\t" + sample['gt'] + "\n"
        s += "PD:\t" + sample["pred"] + "\n"
        s += "Norm of Mean: {}\tNorm of Logvar: {}".format(sample['norm_mean'], sample['norm_logvar'])
        return s

    @staticmethod
    def distance_between_vecs(A, B):
        # TODO
        return LA.norm(A - B)

    @staticmethod
    def read_sample(fname):
        with open(fname, 'r') as fd:
            lines = fd.read().splitlines()

        uniq_id = fname.split("-")[-1].split('.txt')[0]
        uniq_id = int(uniq_id)

        num_of_lines = len(lines)
        assert num_of_lines == 9
        rt = {}
        rt['id'] = uniq_id
        rt['gt'] = lines[1]
        seq_len = len(lines[1].split(" "))
        rt['pred'] = lines[2]
        rt['recon'] = float(lines[3])
        rt['kl'] = float(lines[4]) / seq_len
        rt['nll'] = rt['recon'] + rt['kl']

        rt['code'] = DistAnalyzer.line_to_numpy(lines[6])
        rt['mean'] = DistAnalyzer.line_to_numpy(lines[7])
        rt['norm_mean'] = LA.norm(rt['mean'])
        rt['logvar'] = DistAnalyzer.line_to_numpy(lines[8])
        rt['norm_logvar'] = LA.norm(rt['logvar'])
        return rt


class VMFAnalyzer(DistAnalyzer):
    def __init__(self, path):
        self.path = path

        self.data = self.batch_read_sample(path)
        # For vMF, compare cos of lat code and show 20 sents with close cosine distance
        # implement function of computing inner and inter for cluster
        self.distance_compare()

    @staticmethod
    def distance_compare_unit(data):
        host = data[0]
        host_words = host['gt'].split(" ")
        host_mean = host['mu']
        host_mean = host_mean / LA.norm(host_mean)

        guest = data[1:]
        num_guest = len(guest)
        bag = []
        for idx, g in enumerate(guest):
            g_mean = g['mu']
            g_words = g['gt'].split(" ")
            # jaccard = comp_jaccard_distance(g_words, host_words)
            cos_distance = -1 * (scipy.spatial.distance.cosine(g_mean, host_mean) - 1)
            bag.append([0, cos_distance])
        return bag

    def distance_compare(self, num=50):
        print("Start distance comparison. ")
        all_data_size = len(self.data)
        results = []
        for n in range(num):
            x = random.sample(range(all_data_size), num)
            this_data = [self.data[i] for i in x]
            point = self.distance_compare_unit(this_data)
            results += point
        results = sorted(results, key=lambda x: x[0], reverse=True)

        for r in results:
            # print("{}\t{}".format(r[0], r[1]))
            print((r[1] + 1) // 0.1)
        # print("First num: Jaccard Distance; Second num: Cos distance (smaller closer) (range:0-2)")

    def comp_cos(self):
        pass

    def comp_cos_batch(self):
        pass

    def show_closest_sample(self):
        pass

    def inner_cluster_cos(self):
        pass

    def inter_cluster_cos(self):
        pass

    @staticmethod
    def read_sample(fname):
        with open(fname, 'r') as fd:
            lines = fd.read().splitlines()
        uniq_id = fname.split("-")[1]
        num_of_lines = len(lines)
        assert num_of_lines == 8
        rt = {}
        rt['id'] = uniq_id
        rt['gt'] = lines[1]
        rt['pred'] = lines[2]
        rt['recon'] = float(lines[3])
        rt['kl'] = float(lines[4])
        rt['nll'] = float(lines[5])

        rt['code'] = DistAnalyzer.line_to_numpy(lines[6])
        rt['mu'] = DistAnalyzer.line_to_numpy(lines[7])

        return rt


if __name__ == '__main__':
    # Run Analyzer
    base_path = "/home/cc/save-nvrnn"
    gauss = GaussAnalyzer(os.path.join(base_path,
                                       "Datayelp_Distnor_Modelnvrnn_EnclstmBiFalse_Emb100_Hid400_lat100_lr10.0_drop0.5_kappa0.1_auxw0.0001_normfFalse_nlay1_mixunk1.0_inpzTrue_cdbit50_cdbow200_5.006251241317158logs"))
    vmf = VMFAnalyzer(os.path.join(base_path,
                                   "Datayelp_Distvmf_Modelnvrnn_EnclstmBiFalse_Emb100_Hid400_lat100_lr10.0_drop0.5_kappa100.0_auxw0.0001_normfFalse_nlay1_mixunk1.0_inpzTrue_cdbit50_cdbow200_4.307300597355627logs"))
