import time

import torch

from NVLL.util.util import GVar

from NVLL.util.gpu_flag import GPU_FLAG
print(GPU_FLAG)
#
# start = time.time()
# hard = torch.nn.Hardtanh()
# softmax = torch.nn.Softmax()
# for i in range(100):
#     x = torch.zeros(100000).cuda()
#     y = torch.rand(100000).cuda()
#     z = y * y * y
#     c = y * y / (y + y)
#     d = c * c + c
#     m = y + z + y
#     m = GVar(m)
#
#     for j in range(1000):
#         k = hard(m)
#         e = softmax(m + m)
#         q = softmax(m)
#
# print(time.time() - start)
