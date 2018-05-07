base = "PYTHONPATH=../../ python ../nvll.py --cuda --lr 10.0 --batch_size 20 " \
       "--eval_batch_size 20 --log_interval 500 --model nvrnn --epochs 80  " \
       "--optim sgd --data_name ptb --data_path data/ptb --clip 0.25 " \
       "--input_z --dropout 0.6 --emsize 800 --nhid 800 --aux_weight 0.0001 " \
       " --nlayers 2 --swap 0.0 --replace 0.0  " \
       " "

path_big = " --exp_path /home/cc/exp-nvrnn --root_path /home/cc/vae_txt  "
path_eve = " --exp_path /backup2/jcxu/exp-nvrnn --root_path /home/jcxu/vae_txt  "

base = base + path_big

# RNNLM(zero), nor, vMF
# condition on NA or Bit(20) or BoW(200)
# for vmf, kappa=16,32,64,128 when lat=32

bag = []
for cd_bit in [0]:
    for cd_bow in [0, 200]:
        # for dist in ['zero', 'vmf','nor']:
        for dist in ['vmf', 'nor', 'zero']:
            for lat_dim in [100, 400]:
                for mix_unk in [1]:
                    if dist == 'vmf':
                        if lat_dim == 100:
                            for kappa in [100, 150, 200, 250]:
                                tmp = base + " --cd_bit {} --cd_bow {} --dist {} --kappa {} --mix_unk {} --lat_dim {}". \
                                    format(cd_bit, cd_bow, dist, kappa, mix_unk, lat_dim)
                                bag.append(tmp)
                                print(tmp)
                        elif lat_dim == 400:
                            for kappa in [400, 450, 500, 600, 700]:
                                tmp = base + " --cd_bit {} --cd_bow {} --dist {} --kappa {} --mix_unk {} --lat_dim {}". \
                                    format(cd_bit, cd_bow, dist, kappa, mix_unk, lat_dim)
                                bag.append(tmp)
                                print(tmp)
                        else:
                            raise NotImplementedError
                    else:
                        tmp = base + " --cd_bit {} --cd_bow {} --dist {} --mix_unk {} --lat_dim {}". \
                            format(cd_bit, cd_bow, dist, mix_unk, lat_dim)
                        bag.append(tmp)
                        print(tmp)

                for mix_unk in [0]:
                    if dist == 'vmf':
                        if lat_dim == 100:
                            for kappa in [10, 20, 30]:
                                tmp = base + " --cd_bit {} --cd_bow {} --dist {} --kappa {} --mix_unk {} --lat_dim {}". \
                                    format(cd_bit, cd_bow, dist, kappa, mix_unk, lat_dim)
                                bag.append(tmp)
                                print(tmp)
                        elif lat_dim == 400:
                            for kappa in [10, 20, 50, 100, 200]:
                                tmp = base + " --cd_bit {} --cd_bow {} --dist {} --kappa {} --mix_unk {} --lat_dim {}". \
                                    format(cd_bit, cd_bow, dist, kappa, mix_unk, lat_dim)
                                bag.append(tmp)
                                print(tmp)
                        else:
                            raise NotImplementedError
                    else:
                        tmp = base + " --cd_bit {} --cd_bow {} --dist {} --mix_unk {} --lat_dim {}". \
                            format(cd_bit, cd_bow, dist, mix_unk, lat_dim)
                        bag.append(tmp)
                        print(tmp)
print(len(bag))

cnt_gpu = 4
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
