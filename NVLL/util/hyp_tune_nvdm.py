base = 'PYTHONPATH=../ python nvll.py --cuda --lr 0.005 --batch_size 50 --eval_batch_size 50 --log_interval 200 --model nvdm --epochs 70  --optim adam --data_name rcv --data_path ../data/rcv --clip 1 '

bag = []
for drop in [0., 0.2]:
    for emsize in [128,512]:
        for nhid in [256, 512, 1024]:
            for aux in [0.1, 0.0001]:
                # for dist in ['unifvmf', 'vmf']:
                for dist in ['vmf']:
                    for kappa in [64,128,256]:
                        for lat_dim in [25, 50, 100, 200]:
                            tmp = base+ ' --dropout {} --emsize {} --nhid {} ' \
                                        '--aux_weight {} --dist {} --kappa {} --lat_dim {}' \
                                                    ' '.format\
                                (drop, emsize, nhid, aux, dist, kappa, lat_dim)
                            print(tmp)
                            bag.append(tmp)
print(len(bag))

divid_pieces = 8
import random
random.shuffle(bag)
prints = [[] for _ in range(divid_pieces)]
for idx in range(len(bag)):
    tmp =bag[idx]
    if idx %divid_pieces <2:
        tmp = 'CUDA_VISIBLE_DEVICES=2 '+tmp
    elif idx%divid_pieces<4:
        tmp = 'CUDA_VISIBLE_DEVICES=1 ' + tmp
    else:
        tmp = tmp
    prints[idx%divid_pieces].append(tmp)

cnt = 0
import os
os.chdir('..')
print(os.getcwd())

for p in prints:
    with open('nvdm'+str(cnt)+'.sh','w') as f:
        f.write('\n'.join(p))
    cnt += 1