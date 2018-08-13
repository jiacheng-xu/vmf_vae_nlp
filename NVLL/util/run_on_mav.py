base = "PYTHONPATH=../../ python ../nvll.py --cuda --lr 10.0 --batch_size 20 " \
       "--eval_batch_size 20 --log_interval 500 --model nvrnn --epochs 70  " \
       "--optim sgd --data_name ptb --data_path data/ptb --clip 0.25 " \
       "--input_z --dropout 0.5 --emsize 100 --nhid 400 --aux_weight 0.0001 " \
       " --nlayers 2 --swap 0.0 --replace 0.0  --bi" \
       " "

path_big = " --exp_path /home/cc/exp-nvrnn --root_path /home/cc/vae_txt  "
path_eve = " --exp_path /backup2/jcxu/exp-nvrnn --root_path /home/jcxu/vae_txt  "
path_mav = "--exp_path /work/05235/jcxu/maverick/exp-nvrnn   --root_path /work/05235/jcxu/maverick/vae_txt"
base = base + path_mav

# RNNLM(zero), nor, vMF
# condition on NA or Bit(20) or BoW(200)
# for vmf, kappa=16,32,64,128 when lat=32

bag = []
for cd_bit in [0]:
    for cd_bow in [0]:
        # for dist in ['zero', 'vmf','nor']:
        for dist in ['vmf']:
            for lat_dim in [25, 50, 100]:
                for mix_unk in [1]:
                    if dist == 'vmf':
                        if lat_dim == 100:
                            for kappa in [50, 100, 150, 200]:
                                tmp = base + " --cd_bit {} --cd_bow {} --dist {} --kappa {} --mix_unk {} --lat_dim {}". \
                                    format(cd_bit, cd_bow, dist, kappa, mix_unk, lat_dim)
                                bag.append(tmp)
                                print(tmp)
                        elif lat_dim == 50:
                            for kappa in [25, 50, 100, 150]:
                                tmp = base + " --cd_bit {} --cd_bow {} --dist {} --kappa {} --mix_unk {} --lat_dim {}". \
                                    format(cd_bit, cd_bow, dist, kappa, mix_unk, lat_dim)
                                bag.append(tmp)
                                print(tmp)
                        elif lat_dim == 25:
                            for kappa in [15, 25, 50, 75, 100]:
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

                # for mix_unk in [0]:
                #     if dist == 'vmf':
                #         if lat_dim == 25:
                #             for kappa in [5, 10]:
                #                 tmp = base + " --cd_bit {} --cd_bow {} --dist {} --kappa {} --mix_unk {} --lat_dim {}". \
                #                     format(cd_bit, cd_bow, dist, kappa, mix_unk, lat_dim)
                #                 bag.append(tmp)
                #                 print(tmp)
                #         # if lat_dim == 100:
                #         #     for kappa in [10, 20, 30]:
                #         #         tmp = base + " --cd_bit {} --cd_bow {} --dist {} --kappa {} --mix_unk {} --lat_dim {}". \
                #         #             format(cd_bit, cd_bow, dist, kappa, mix_unk, lat_dim)
                #         #         bag.append(tmp)
                #         #         print(tmp)
                #         # elif lat_dim == 50:
                #         #     for kappa in [5, 10, 20,  30]:
                #         #         tmp = base + " --cd_bit {} --cd_bow {} --dist {} --kappa {} --mix_unk {} --lat_dim {}". \
                #         #             format(cd_bit, cd_bow, dist, kappa, mix_unk, lat_dim)
                #         #         bag.append(tmp)
                #         #         print(tmp)
                #         else:
                #             raise NotImplementedError
                #     else:
                #         tmp = base + " --cd_bit {} --cd_bow {} --dist {} --mix_unk {} --lat_dim {}". \
                #             format(cd_bit, cd_bow, dist, mix_unk, lat_dim)
                #         bag.append(tmp)
                #         print(tmp)
print(len(bag))

divid_pieces = len(bag)

import random

random.shuffle(bag)
prints = [[] for _ in range(divid_pieces)]
for idx in range(len(bag)):
    tmp = bag[idx]

    tmp = 'CUDA_VISIBLE_DEVICES=0 ' + tmp
    prints[idx % divid_pieces].append(tmp)

cnt = 0
import os

# os.chdir('..')
print(os.getcwd())

for p in prints:
    with open('rrnvrnn' + str(cnt) + '.sh', 'w') as f:
        f.write("#!/bin/sh\n#SBATCH -n 1\n#SBATCH -p gpu\n#SBATCH -A Handholding-Latent-V"
                "\n#SBATCH -N 1\n#SBATCH -o out{}.txt\n#SBATCH -e err{}.txt"
                "\n#SBATCH -t 12:00:00\nsource /work/05235/jcxu/maverick/anaconda3/etc/profile.d/conda.sh"
                "\nconda deactivate\nconda activate\n".format(str(cnt), str(cnt)))
        # f.write(
        #     "trap '{ echo \"Hey, you pressed Ctrl-C. Press Ctrl-D or Kill this screen to kill this screen.\" ; }' INT\n")
        f.write("cd /work/05235/jcxu/maverick/vae_txt/NVLL/util\n")
        f.write('\n'.join(p))
    cnt += 1
