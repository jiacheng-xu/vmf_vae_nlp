base = 'PYTHONPATH=../../ python ../nvll.py --cuda ' \
       '--lr 1 --batch_size 50 --eval_batch_size 50' \
       ' --log_interval 400 --model nvdm --epochs 100  ' \
       '--optim sgd  --clip 1 ' \
       '--data_path data/rcv --data_name rcv ' \
       ' --dist vmf'

path_big = " --exp_path /home/cc/exp-nvdm --root_path /home/cc/vae_txt  "
path_eve = " --exp_path /backup2/jcxu/exp-nvdm --root_path /home/jcxu/vae_txt  "

base = base + path_eve

bag = []
for drop in [0.1]:
    for emsize in [400]:
        for nhid in [800]:
            for aux in [0.0001]:
                # for dist in ['unifvmf', 'vmf']:
                for dist in ['vmf']:
                    for lat_dim in [50]:
                        if lat_dim == 25:
                            for kappa in [50, 100]:
                                tmp = base + ' --dropout {} --emsize {} --nhid {} ' \
                                             '--aux_weight {} --dist {} --kappa {} --lat_dim {}' \
                                             ' '.format \
                                    (drop, emsize, nhid, aux, dist, kappa, lat_dim)
                                print(tmp)
                                bag.append(tmp)
                        elif lat_dim == 50:
                            for kappa in [150]:
                                tmp = base + ' --dropout {} --emsize {} --nhid {} ' \
                                             '--aux_weight {} --dist {} --kappa {} --lat_dim {}' \
                                             ' '.format \
                                    (drop, emsize, nhid, aux, dist, kappa, lat_dim)
                                print(tmp)
                                bag.append(tmp)
                        elif lat_dim == 200:
                            for kappa in [50, 100, 150]:
                                tmp = base + ' --dropout {} --emsize {} --nhid {} ' \
                                             '--aux_weight {} --dist {} --kappa {} --lat_dim {}' \
                                             ' '.format \
                                    (drop, emsize, nhid, aux, dist, kappa, lat_dim)
                                print(tmp)
                                bag.append(tmp)

print(len(bag))

cnt_gpu = 4
per_gpu = 1
divid_pieces = cnt_gpu * per_gpu

import random

random.shuffle(bag)
prints = [[] for _ in range(divid_pieces)]
for idx in range(len(bag)):
    tmp = bag[idx]
    if idx % divid_pieces < per_gpu:
        tmp = 'CUDA_VISIBLE_DEVICES=0 ' + tmp
    elif idx % divid_pieces < per_gpu * 2:
        tmp = 'CUDA_VISIBLE_DEVICES=1 ' + tmp
    elif idx % divid_pieces < per_gpu * 3:
        tmp = 'CUDA_VISIBLE_DEVICES=2 ' + tmp
    else:
        tmp = 'CUDA_VISIBLE_DEVICES=3 ' + tmp
    prints[idx % divid_pieces].append(tmp)

cnt = 0
import os

# os.chdir('..')
print(os.getcwd())

for p in prints:
    with open('dm' + str(cnt) + '.sh', 'w') as f:
        f.write(
            "trap '{ echo \"Hey, you pressed Ctrl-C. Press Ctrl-D or Kill this screen to kill this screen.\" ; }' INT\n")

        f.write('\n'.join(p))
    cnt += 1
