base = 'PYTHONPATH=../../ python ../nvll.py --cuda ' \
       '--lr 0.01 --batch_size 50 --eval_batch_size 50' \
       ' --log_interval 50 --model nvdm --epochs 50  ' \
       '--optim adam  --clip 2 ' \
       '--data_path data/20ng --data_name 20ng ' \
       '--exp_path /home/cc/exp-nvdm --dist vmf'
bag = []
for drop in [0., 0.2]:
    for emsize in [100,400]:
        for nhid in [100, 400]:
            for aux in [0.1, 0.0001]:
                # for dist in ['unifvmf', 'vmf']:
                for dist in ['vmf']:
                    for kappa in [25,50,200]:
                        for lat_dim in [25, 50,  200]:
                            tmp = base+ ' --dropout {} --emsize {} --nhid {} ' \
                                        '--aux_weight {} --dist {} --kappa {} --lat_dim {}' \
                                                    ' '.format\
                                (drop, emsize, nhid, aux, dist, kappa, lat_dim)
                            print(tmp)
                            bag.append(tmp)
print(len(bag))


cnt_gpu = 4
per_gpu = 3
divid_pieces = cnt_gpu * per_gpu

import random
random.shuffle(bag)
prints = [[] for _ in range(divid_pieces)]
for idx in range(len(bag)):
    tmp =bag[idx]
    if idx %divid_pieces <per_gpu:
        tmp = 'CUDA_VISIBLE_DEVICES=0 '+tmp
    elif idx%divid_pieces<per_gpu*2:
        tmp = 'CUDA_VISIBLE_DEVICES=1 ' + tmp
    elif idx%divid_pieces<per_gpu*3:
        tmp = 'CUDA_VISIBLE_DEVICES=2 ' + tmp
    else:
        tmp = 'CUDA_VISIBLE_DEVICES=3 ' + tmp
    prints[idx%divid_pieces].append(tmp)

cnt = 0
import os
# os.chdir('..')
print(os.getcwd())

for p in prints:
    with open('nvdm'+str(cnt)+'.sh','w') as f:
        f.write('\n'.join(p))
    cnt += 1