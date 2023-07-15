import random
import numpy as np
import torch
import os
import json

def ready_pretrain_data(path):
    with open(file=os.path.join(path,'train_data.txt'),encoding='utf-8',mode='r') as f:
        data = [line.strip() for line in f if line.strip()]
    return data

def set_seed(seed):
    """
    设置随机种子
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)



