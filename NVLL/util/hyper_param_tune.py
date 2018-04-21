base = 'PYTHONPATH=../ python nvll.py --cuda --lr 0.005 --batch_size 50 --eval_batch_size 50 --log_interval 200 --model nvdm --epochs 70  --optim adam --tied --data_name 20news --data_path ../data/20news --clip 5 '

bag = []
for drop in [0.2, 0.5]:
    for emsize in [100, 400]:
        for nhid in [100, 400]:
            for aux in [0.1, 0.0001]:
                for dist in ['unifvmf', 'vmf']:
                    for kappa in [32, 64,128,256]:
                        for lat_dim in [32, 64, 128, 256]:
                            tmp = base+ '--dropout {} --emsize {} --nhid {} ' \
                                        '--aux_weight {} --dist {} --kappa {} --lat_dim {}'.format\
                                (drop, emsize, nhid, aux, dist, kappa, lat_dim)
                            # print(tmp)
                            bag.append(tmp)
print(len(bag))

import random
random.shuffle(bag)
prints = [[] for _ in range(8)]
for idx in range(len(bag)):
    tmp =bag[idx]
    if idx %8 <2:
        tmp = 'CUDA_VISIBLE_DEVICES=2 '+tmp
    else:
        tmp = 'CUDA_VISIBLE_DEVICES=1 ' + tmp
    prints[idx%8].append(tmp)

cnt = 0
import os
os.chdir('..')
print(os.getcwd())

for p in prints:
    with open(str(cnt)+'.sh','w') as f:
        f.write('\n'.join(p))
    cnt += 1