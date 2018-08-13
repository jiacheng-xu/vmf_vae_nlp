base = "PYTHONPATH=../../ python ../nvll.py --cuda --lr 10.0 --batch_size 20 " \
       "--eval_batch_size 20 --log_interval 500 --model nvrnn --epochs 80  " \
       "--optim sgd --data_name ptb --data_path data/ptb --clip 0.25 " \
       "--input_z --dropout 0.5 --emsize 100 --aux_weight 0.0001 " \
       "  --swap 0.0 --replace 0.0  " \
       " "

path_eve = " --exp_path /backup2/jcxu/exp-nvrnn --root_path /home/jcxu/vae_txt  "
base = base + path_eve

# RNNLM(zero), nor, vMF
# condition on NA or Bit(20) or BoW(200)
# for vmf, kappa=16,32,64,128 when lat=32

bag = []
# for enc in ['bow','gru','lstm']:
#     for dec in ['  --nlayers 1 --nhid 400 ','  --nlayers 3 --nhid 400 ' ]:
#         for anneal in [0,2]:
#             for mix_unk in [0]:
#                 tmp = base + " --lat_dim 50  --enc_type {}  --dist nor   {}   --mix_unk {} --anneal {}". \
#                     format(enc, dec, mix_unk, anneal)
#                 bag.append(tmp)
#                 print(tmp)

for enc in ['bow', 'gru', 'lstm']:
    for dec in ['  --nlayers 1 --nhid 400 ', '  --nlayers 3 --nhid 400 ']:
        for anneal in [0]:
            for mix_unk in [1]:
                for kappa in [15, 30, 45, 60, 75]:
                    tmp = base + "--lat_dim 15 --enc_type {}  --dist vmf  {}   --mix_unk {} --anneal {} --kappa {}". \
                        format(enc, dec, mix_unk, anneal, kappa)
                    bag.append(tmp)
                    print(tmp)

print(len(bag))

cnt_gpu = 3
per_gpu = 2
divid_pieces = cnt_gpu * per_gpu

import random

random.shuffle(bag)
prints = [[] for _ in range(divid_pieces)]
for idx in range(len(bag)):
    tmp = bag[idx]

    # N = random.randrange(0, cnt_gpu)
    # tmp = 'CUDA_VISIBLE_DEVICES={} '.format(N) + tmp
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
    with open('rrnvrnn' + str(cnt) + '.sh', 'w') as f:
        f.write(
            "trap '{ echo \"Hey, you pressed Ctrl-C. Press Ctrl-D or Kill this screen to kill this screen.\" ; }' INT\n")
        f.write('\n'.join(p))
    cnt += 1
