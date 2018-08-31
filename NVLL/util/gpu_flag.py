import torch

use_gpu = True
if torch.cuda.is_available() and use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
