# 【分布式训练】单机多卡的正确打开方式（三）：PyTorch

 http://fyubang.com/2019/07/23/distributed-training3/

Posted on 2019-07-23 | In [训练方法 ](http://fyubang.com/categories/训练方法/), [分布式训练 ](http://fyubang.com/categories/训练方法/分布式训练/)| 842

拖更拖更了，今天讲一下PyTorch下要如何单机多卡训练。

PyTorch的数据并行相对于TensorFlow而言，要简单的多，主要分成两个API：

- DataParallel（DP）：Parameter Server模式，一张卡位reducer，实现也超级简单，一行代码。
- DistributedDataParallel（DDP）：All-Reduce模式，本意是用来分布式训练，但是也可用于单机多卡。

## 1. DataParallel

DataParallel是基于Parameter server的算法，负载不均衡的问题比较严重，有时在模型较大的时候（比如bert-large），reducer的那张卡会多出3-4g的显存占用。

先简单定义一下数据流和模型。

```
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os

input_size = 5
output_size = 2
batch_size = 30
data_size = 30

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("  In Model: input size", input.size(),
              "output size", output.size())
        return output
model = Model(input_size, output_size)

if torch.cuda.is_available():
    model.cuda()
    
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 就这一行
    model = nn.DataParallel(model)
    
for data in rand_loader:
    if torch.cuda.is_available():
        input_var = Variable(data.cuda())
    else:
        input_var = Variable(data)
    output = model(input_var)
    print("Outside: input size", input_var.size(), "output_size", output.size())
```

## 2. DDP

官方建议用新的DDP，采用all-reduce算法，本来设计主要是为了多机多卡使用，但是单机上也能用，使用方法如下：

初始化使用nccl后端

```
torch.distributed.init_process_group(backend="nccl")
```



模型并行化

```
model=torch.nn.parallel.DistributedDataParallel(model)
```



需要注意的是：DDP并不会自动shard数据

1. 如果自己写数据流，得根据`torch.distributed.get_rank()`去shard数据，获取自己应用的一份

2. 如果用Dataset API，则需要在定义Dataloader的时候用

   ```
   DistributedSampler
   ```

    

   去shard：

   ```
   sampler = DistributedSampler(dataset) # 这个sampler会自动分配数据到各个gpu上
   DataLoader(dataset, batch_size=batch_size, sampler=sampler)
   ```

完整的例子：

```
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data.distributed import DistributedSampler
# 1) 初始化
torch.distributed.init_process_group(backend="nccl")

input_size = 5
output_size = 2
batch_size = 30
data_size = 90

# 2） 配置每个进程的gpu
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size).to('cuda')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

dataset = RandomDataset(input_size, data_size)
# 3）使用DistributedSampler
rand_loader = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         sampler=DistributedSampler(dataset))

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("  In Model: input size", input.size(),
              "output size", output.size())
        return output
    
model = Model(input_size, output_size)

# 4) 封装之前要把模型移到对应的gpu
model.to(device)
    
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 5) 封装
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
   
for data in rand_loader:
    if torch.cuda.is_available():
        input_var = data
    else:
        input_var = data

    output = model(input_var)
    print("Outside: input size", input_var.size(), "output_size", output.size())
```



需要通过命令行启动

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 torch_ddp.py
```

结果：

```
Let's use 2 GPUs!
Let's use 2 GPUs!
  In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
  In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([15, 5]) output_size torch.Size([15, 2])
  In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
  In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([15, 5]) output_size torch.Size([15, 2])
```



可以看到有两个进程，log打印了两遍

torch.distributed.launch 会给模型分配一个args.local_rank的参数，也可以通过torch.distributed.get_rank()获取进程id