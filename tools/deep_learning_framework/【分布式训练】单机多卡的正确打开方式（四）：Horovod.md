# 【分布式训练】单机多卡的正确打开方式（四）：Horovod

 http://fyubang.com/2019/07/26/distributed-training4/

Posted on 2019-07-26 | In [训练方法 ](http://fyubang.com/categories/训练方法/), [分布式训练 ](http://fyubang.com/categories/训练方法/分布式训练/)| 427

讲完了单机多卡的分布式训练的理论、TensorFlow和PyTorch分别的实现后，今天瓦砾讲一个强大的第三方插件：Horovod。

Horovod是Uber开源的跨平台的分布式训练工具，名字来自于俄国传统民间舞蹈，舞者手牵手围成一个圈跳舞，与Horovod设备之间的通信模式很像，有以下几个特点：

1. 兼容TensorFlow、Keras和PyTorch机器学习框架。
2. 使用Ring-AllReduce算法，对比Parameter Server算法，有着无需等待，负载均衡的优点。
3. 实现简单，五分钟包教包会。（划重点）

Uber官方在git上给了很详细的例子： https://github.com/horovod/horovod/tree/master/examples，所以这里只简单讲一下大概的使用方法：

## TensorFlow

以TF的Custom Training Loop API为例：

```
import tensorflow as tf
import horovod.tensorflow as hvd

# 1. 初始化horovod
hvd.init()
# 2. 给当前进程分配对应的gpu，local_rank()返回的是当前是第几个进程
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())
# 3. Scale学习率，封装优化器
opt = tf.train.AdagradOptimizer(0.01 * hvd.size())
opt = hvd.DistributedOptimizer(opt)
# 4. 定义初始化的时候广播参数的hook，这个是为了在一开始的时候同步各个gpu之间的参数
hooks = [hvd.BroadcastGlobalVariablesHook(0)]
# 搭建model，定义loss
loss = ...
train_op = opt.minimize(loss)
# 5. 只保存一份ckpt就行
checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None
# 7. 用MonitoredTrainingSession实现初始化，读写ckpt
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)
```



具体的代码看`tensorflow_mnist.py`：https://github.com/horovod/horovod/blob/master/examples/tensorflow_mnist.py

单机双卡训练输入以下命令：

```
CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 -H localhost:2 python tensorflow_mnist.py
```



这里 `-np`指的是进程的数量。

执行之后可以看到如下的结果，因为多线程，每个step都打印了两遍。

```
[1,0]<stderr>:INFO:tensorflow:loss = 0.13126025, step = 300 (0.191 sec)
[1,1]<stderr>:INFO:tensorflow:loss = 0.01396352, step = 310 (0.177 sec)
[1,0]<stderr>:INFO:tensorflow:loss = 0.063738815, step = 310 (0.182 sec)
[1,1]<stderr>:INFO:tensorflow:loss = 0.044452004, step = 320 (0.215 sec)
[1,0]<stderr>:INFO:tensorflow:loss = 0.028987963, step = 320 (0.212 sec)
[1,0]<stderr>:INFO:tensorflow:loss = 0.09094897, step = 330 (0.206 sec)
[1,1]<stderr>:INFO:tensorflow:loss = 0.11366991, step = 330 (0.210 sec)
[1,0]<stderr>:INFO:tensorflow:loss = 0.08559138, step = 340 (0.200 sec)
[1,1]<stderr>:INFO:tensorflow:loss = 0.037002128, step = 340 (0.201 sec)
[1,0]<stderr>:INFO:tensorflow:loss = 0.15422738, step = 350 (0.181 sec)
[1,1]<stderr>:INFO:tensorflow:loss = 0.06424393, step = 350 (0.179 sec)
```

## PyTorch

Torch下也是类似的套路，但是由于PyTorch本身单机多卡训练已经够简单了，API也稳定，所以笔者一般做的时候就是直接用Torch自己的`DP`和`DDP`了。

```
import torch
import horovod.torch as hvd

# 1. 初始化horovod
hvd.init()
# 2. 给当前进程分配对应的gpu，local_rank()返回的是当前是第几个进程
torch.cuda.set_device(hvd.local_rank())
# Define dataset...
train_dataset = ...
# 3. 用DistributedSampler给各个worker分数据
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)
# Build model...
model = ...
model.cuda()
# 4. 封装优化器
optimizer = optim.SGD(model.parameters())
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
# 5. 初始化的时候广播参数，这个是为了在一开始的时候同步各个gpu之间的参数
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
# 训练
for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
       optimizer.zero_grad()
       output = model(data)
       loss = F.nll_loss(output, target)
       loss.backward()
       optimizer.step()
       if batch_idx % args.log_interval == 0:
           print('Train Epoch: {} [{}/{}]\tLoss: {}'.format(
               epoch, batch_idx * len(data), len(train_sampler), loss.item()))
```

## 速度

瓦砾还没有来得及做一个全面的Horovod、`tf.distribute`和 Torch的单机多卡训练速度的横向对比，不过大家可以参考这两篇：

1. [Horovod: fast and easy distributed deep learning in TensorFlow](https://arxiv.org/pdf/1802.05799.pdf)
2. [Goodbye Horovod, Hello CollectiveAllReduce](https://www.logicalclocks.com/goodbye-horovod-hello-tensorflow-collectiveallreduce/)

总体而言，用了All-Reduce算法的API，速度应该都差不多，如果你是土豪，拥有NVLINK（卡间通信极快）的话，那忘了我说的这几篇“废话”吧朋友。Orz。

## 总结

终于结束了单机多卡系列的最后一章，由于博客本身的限制，给的例子整体还是比较简单，以入门为主，大家具体使用的时候肯定还是会遇到一些坑，这里瓦砾把踩过的一些坑和解决办法列举在这，以避免大家以后重复踩坑：

- tf.contrib.distributed.MirroredStrategy 需要optimizer支持merge_call（bert实现的optimizer是直接修改apply_gradient的，所以会报错），这个时候就需要正确地修改optimizer里的_apply_dense、_apply_sparse(参考[Issue 23986](https://github.com/tensorflow/tensorflow/issues/23986) 和 [JayYip](https://github.com/JayYip/bert-multitask-learning/blob/master/bert_multitask_learning/optimizer.py))。或者用horovod，就可以避免这个问题。
- Effective batch size，不同的多卡工具对输入的batch size的操作不一样，要确定最后进模型的effective batch size才有意义。一般来说，多进程的batch size指的是每张卡的batch size。
- Learning rate scale，学习率要根据[effective batch size](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md)调整。
- All-Reduce由于是多进程的，数据流各自独立，为了防止同一个step多gpu的batch重叠，最好的的办法是在每个进程里根据local_rank设置shard的数据，保证各个gpu采样的数据不重叠。
- 为了使用horovod，新建docker container时，要加—privileged，否则会[疯狂报warning](https://github.com/horovod/horovod/issues/653)，虽然没影响，但是看着难受。
- Pytorch的DP多卡要注意最后一个batch的batch size不能小于gpu的数量，否则会报错，最保险的做法是drop_last，扔掉最后的batch。
- 并不是所有情况下All-Reduce都比PS好，比如当卡间通信用的是NVLink的时候，在gpu数量不多的情况下，数据传输的时间不是瓶颈，All-Reduce的提升就几乎没有了。
- DP和DDP有一个[区别](https://discuss.pytorch.org/t/training-performance-degrades-with-distributeddataparallel/47152)在于BatchNorm。
- DDP封装model后不能再改动model。
- 待补充。。。

## Reference

1. [Horovod的官方给的一些例子](https://github.com/horovod/horovod/tree/master/examples)。
2. [Uber：如何用Horovod实现bert的单机多卡训练](https://lambdalabs.com/blog/bert-multi-gpu-implementation-using-tensorflow-and-horovod-with-code/)
3. [Goodbye Horovod, Hello CollectiveAllReduce](https://www.logicalclocks.com/goodbye-horovod-hello-tensorflow-collectiveallreduce/)
4. [Horovod: fast and easy distributed deep learning in TensorFlow](https://arxiv.org/pdf/1802.05799.pdf)