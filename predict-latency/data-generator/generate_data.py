import numpy as np
import os
import argparse

from models import *
from calculate_latency import calculate_latency

# bucket 안 row 개수
num_data = 100


def generate_data():

    expansion = np.random.randint(1, 7, size=(num_data,1))
    out_planes = np.random.randint(16, 321, size=(num_data,1))
    num_blocks = np.random.randint(1, 16, size=(num_data,1))
    stride = np.random.randint(1, 5, size=(num_data,1))

    bucket = np.hstack([expansion, out_planes, num_blocks, stride])
    # print(bucket)

    num_types = np.random.randint(1, 10)
    x_data = bucket[np.random.choice(bucket.shape[0], 7, replace=False)]
    # print(x_data)
    cfg = x_data.tolist()
    cfg = list(tuple(e) for e in cfg)

    target = calculate_latency(cfg)

    # print(cfg)
    # x_tensor = torch.from_numpy(x_data)
    # print(x_tensor)

    print(cfg)
    print(target)

    # 직렬화
    serialized_cfg = []
    for e in cfg:
        for a in e:
            serialized_cfg.append(str(a) + ' ')

    print("return ~")
    return serialized_cfg, str(target)







### convert from float64 to float32 for various reasons:
### speedup, less memory usage, precision is enough.
### when using GPU, fp16, fp32 or fp64 depends on
### type of GPU (consumer/workstation/server).
