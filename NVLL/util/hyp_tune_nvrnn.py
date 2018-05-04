base = "PYTHONPATH=../../ python ../nvll.py --cuda --lr 10.0 --batch_size 50 " \
       "--eval_batch_size 50 --log_interval 300 --model nvrnn --epochs 150  " \
       "--optim sgd --data_name yelp --data_path data/yelp --clip 0.25 " \
       "--input_z --dropout 0.5 --emsize 100 --nhid 400 --aux_weight 0.0001 " \
       "--lat_dim 100 --nlayers 1 --swap 0.0 --replace 0.0  " \
       "--mix_unk 0  "

path_big = " --exp_path /home/cc/exp-nvrnn --root_path /home/cc/vae_txt  "
path_eve = " --exp_path /backup2/jcxu/exp-nvrnn --root_path /home/jcxu/vae_txt  "

base = base + path_eve

# RNNLM(zero), nor, vMF
# condition on NA or Bit(20) or BoW(200)
# for vmf, kappa=16,32,64,128 when lat=32

bag = []
for cd_bit in [50]:
    for cd_bow in [0, 200]:
        # for dist in ['zero', 'vmf','nor']:
        for dist in [ 'vmf']:
            for lat_dim in [100]:
                if dist == 'vmf':
                    for kappa in [50,100,200]:
                        tmp = base +" --cd_bit {} --cd_bow {} --dist {} --kappa {}".format(cd_bit, cd_bow,dist,kappa)
                        bag.append(tmp)
                else:
                    tmp = base +" --cd_bit {} --cd_bow {} --dist {}".format(cd_bit, cd_bow,dist)
                    bag.append(tmp)

print(len(bag))

cnt_gpu = 2
per_gpu = 3
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
    with open('nvrnn' + str(cnt) + '.sh', 'w') as f:
        f.write("trap '{ echo \"Hey, you pressed Ctrl-C. Press Ctrl-D or Kill this screen to kill this screen.\" ; }' INT\n")
        f.write('\n'.join(p))
    cnt += 1
