base = 'PYTHONPATH=../ python nvll.py --cuda --lr 10 --batch_size 50 --eval_batch_size 50 --log_interval 200 --model nvrnn --epochs 70  --optim sgd --data_name ptb --data_path ../data/ptb --clip 0.25 '

bag = []
for drop in [0.5, 0.7]:
    for emsize in [100]:
        for nhid in [400, 800]:
            for aux in [0.1, 0.0001]:
                # for dist in ['unifvmf', 'vmf']:
                for dist in ['vmf']:
                    for kappa in [8, 16, 32]:
                        for lat_dim in [32, 64, 128, 256]:
                            for mix_unk in [0.0, 0.25]:
                                for nlayers in [1, 2]:
                                    tmp = base + '--input_z --dropout {} --emsize {} --nhid {} ' \
                                                 '--aux_weight {} --dist {} --kappa {} --lat_dim {}' \
                                                 ' --mix_unk {} --nlayers {}'.format \
                                        (drop, emsize, nhid, aux, dist, kappa, lat_dim, mix_unk, nlayers)
                                    print(tmp)
                                    bag.append(tmp)
print(len(bag))

divid_pieces = 8
import random

random.shuffle(bag)
prints = [[] for _ in range(divid_pieces)]
for idx in range(len(bag)):
    tmp = bag[idx]
    if idx % divid_pieces < 2:
        tmp = 'CUDA_VISIBLE_DEVICES=2 ' + tmp
    elif idx % divid_pieces < 4:
        tmp = 'CUDA_VISIBLE_DEVICES=1 ' + tmp
    else:
        tmp = tmp
    prints[idx % divid_pieces].append(tmp)

cnt = 0
import os

os.chdir('..')
print(os.getcwd())

for p in prints:
    with open('nvrnn' + str(cnt) + '.sh', 'w') as f:
        f.write('\n'.join(p))
    cnt += 1
