import numpy as np
import os
import argparse

from models import *
from calculate_latency import calculate_latency

"""

# (expansion, out_planes, num_blocks, stride)
cfg = [(1,  16, 1, 2),
       (6,  24, 2, 1),
       (6,  40, 2, 2),
       (6,  80, 3, 2),
       (6, 112, 3, 1),
       (6, 192, 4, 2),
       (6, 320, 1, 2)]

num_types: 1~15 integer  !아래 예제는 7! (input dimension 고정을 위햬)
---
expansion: 1~6 integer
out_planes: 16~320 integer
num_blocks: 1~20 integer
stride: 1 ~ 8 integer

assume that depth scale is fixed.

"""
# bucket 안 row 개수
num_data = 10


def generate_data(testloader):

    np.random.seed()
    expansion = np.random.randint(1, 7, size=(num_data,1))
    out_planes = np.random.randint(16, 321, size=(num_data,1))
    num_blocks = np.random.randint(1, 21, size=(num_data,1))
    stride = np.random.randint(1, 9, size=(num_data,1))

    bucket = np.hstack([expansion, out_planes, num_blocks, stride])
    print(bucket)

    num_types = np.random.randint(1, 10)
    x_data = bucket[np.random.choice(bucket.shape[0], 7, replace=False)]
    print(x_data)
    cfg = x_data.tolist()
    cfg = list(tuple(e) for e in cfg)

    target = calculate_latency(cfg, testloader)


    # print(cfg)
    # x_tensor = torch.from_numpy(x_data)
    # print(x_tensor)

    # print("in generate_data()")
    # print(cfg)
    # print(target)

    # 직렬화
    serialized_cfg = []
    for e in cfg:
        for a in e:
            serialized_cfg.append(str(a) + ' ')

    # print("return ~")
    return serialized_cfg, str(target)





