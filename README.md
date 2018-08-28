Spherical Latent Spaces for Stable Variational Autoencoders (vMF-VAE)
=======================

In this repo, we provide the experimental setups and inplementation for the algorithms described in:

    Spherical Latent Spaces for Stable Variational Autoencoders.
    Jiacheng Xu and Greg Durrett. EMNLP 2018.
    
Please cite:

    ??
    
## About

Keyword: **PyTorch**, **VAE**, **NLP**
What to get from this repo: 
* Original **Gaussian VAE** with tuned hyper-parameters and pre-trained models;
* Novel **von-Mises Fisher VAE (vMF-VAE)** with tuned hyper-parameters and pre-trained models.

## Setup
The environment base is Python 3.6 and Anaconda.

The codes are originally developed in PyTorch 0.3.1 and upgraded to PyTorch 0.4.1.

    conda install pytorch=0.4.1 torchvision -c pytorch
    pip install tensorboardX

### Data

#### Data for Document Model
In this paper, we use the exact same pre-processed dataset, 20NG and RC, as Miao et al. used in 
[Neural Variational Inference for Text Processing](https://arxiv.org/abs/1511.06038). Here is the [link to Miao's repo](https://github.com/ysmiao/nvdm).
* [Download RC](https://utexas.box.com/s/36iue908zi0m41ee4ciy8e2xi48bcwko) (Email me or submit an issue if it doesn't work)
* Location of 20 News Group(20ng): `data/20ng`.

####Data for Language Model
We use the standard PTB and Yelp. Datasets are included in `data`.

## Running

#### Set up Device: CUDA or CPU

The choice of cpu or gpu can be modified at `NVLL/util/gpu_flag.py`.


### Train and Test

#### Training Neural Variational Document Model (NVDM)
    
    cd NVLL
    # Training vMF VAE on 20 News group
    PYTHONPATH=../ python nvll.py --lr 1 --batch_size 50 --eval_batch_size 50 --log_interval 75 --model nvdm --epochs 100  --optim sgd  --clip 1 --data_path data/20ng --data_name 20ng  --dist vmf --exp_path /backup2/jcxu/exp-nvdm --root_path /home/jcxu/vae_txt   --dropout 0.1 --emsize 100 --nhid 400 --aux_weight 0.0001 --dist vmf --kappa 100 --lat_dim 25
    
    # Training vMF VAE on RC
    PYTHONPATH=../ python nvll.py --lr 1 --batch_size 50 --eval_batch_size 50 --log_interval 1000 --model nvdm --epochs 100  --optim sgd  --clip 1 --data_path data/rcv --data_name rcv  --dist vmf --exp_path /backup2/jcxu/exp-nvdm --root_path /home/jcxu/vae_txt   --dropout 0.1 --emsize 400 --nhid 800 --aux_weight 0.0001 --dist vmf --kappa 150 --lat_dim 50

#### Training Neural Variational Recurrent Language Model (NVRNN)
     

## Reference


## Contact
Submit an issue here or find more information in my [homepage](http://www.cs.utexas.edu/~jcxu/).
