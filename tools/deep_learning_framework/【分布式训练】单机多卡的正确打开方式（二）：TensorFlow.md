# 【分布式训练】单机多卡的正确打开方式（二）：TensorFlow

 http://fyubang.com/2019/07/14/distributed-training2/

Posted on 2019-07-14 | In [训练方法 ](http://fyubang.com/categories/训练方法/), [分布式训练 ](http://fyubang.com/categories/训练方法/分布式训练/)| 616

瓦砾上一篇讲了单机多卡分布式训练的一些入门介绍，后面几篇准备给大家讲讲TensorFlow、PyTorch框架下要怎么实现多卡训练。

这一篇就介绍一下TensorFlow上的分布式训练，尽管从传统的Custom Training Loops到Estimator再到Keras，TF的API换来换去让人猝不及防、又爱又恨，但是由于种种原因，TensorFlow还是业务上最成熟的框架，所以Let’s还是do it。（没看过上一篇的读者建议先看一下原理部分：[分布式训练的正确打开方式（一）：理论基础](http://fyubang.com/2019/07/08/distributed-training/)，因为算法的理论理解对于后面API的理解还是很重要的。）

这篇博客主要介绍TensorFlow在1.13版本里发布的`tf.distribute` API，集成了之前`tf.contrib.distribute`的很多功能，并且大大的简化了使用。官方很良心的放了Google Colab，想要一步步执行看结果的读者可以移步[官方教学](https://www.tensorflow.org/guide/distribute_strategy)。

## Overview

`tf.distribute`的核心API是`tf.distribute.Strategy`，可以简简单单几行代码就实现单机多卡，多机多卡等情况的分布式训练。主要有这几个优势：

- 简单易用，开箱即用，高性能。
- 便于各种分布式Strategy切换。
- 支持Custom Training Loop、Estimator、Keras。
- 支持eager excution。

```
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
```

## Strategy的类别

`tf.distribute.Strategy`设计的初衷是能cover不同维度的use cases，目前主要有四个Strategy：

- MirroredStrategy
- CentralStorageStrategy
- MultiWorkerMirroredStrategy
- ParameterServerStrategy

还有一些策略，例如异步训练等等，后面会逐步支持。

### 1. MirroredStrategy

镜像策略用于单机多卡 数据并行 同步更新的情况，在每个GPU上保存一份模型副本，模型中的每个变量都镜像在所有副本中。这些变量一起形成一个名为`MirroredVariable`的概念变量。通过apply相同的更新，这些变量保持彼此同步。

镜像策略用了高效的All-reduce算法来实现设备之间变量的传递更新。默认情况下，它使用NVIDIA NCCL作为all-reduce实现。用户还可以在官方提供的其他几个选项之间进行选择。

最简单的创建一个镜像策略的方法：

```
mirrored_strategy = tf.distribute.MirroredStrategy()
```

也可以自己定义要用哪些devices：

```
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
```

官方也提供了其他的一些all-reduce实现：

- `tf.distribute.CrossDeviceOps`
- `tf.distribute.HierarchicalCopyAllReduce`
- `tf.distribute.ReductionToOneDevice`
- `tf.distribute.NcclAllReduce` (default)

```
mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
```

### 2. CentralStorageStrategy

中央存储策略，参数被统一存在CPU里，然后复制到所有GPU上，优点是GPU负载均衡了，但是一般情况下CPU和GPU通信代价大，不建议使用。

```
central_storage_strategy = tf.distribute.experimental.CentralStorageStrategy()
```

### 3. MultiWorkerMirroredStrategy

这个API和`MirroredStrategy`很类似，是其多机多卡分布式的版本，由于我们主要是介绍单机多卡，这里就不展开讲了。

```
multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
```

### 4. ParameterServerStrategy

这个API呢，就是被大家普遍嫌弃即将淘汰的PS策略，慢+负载不均衡。（和all-reduce的区别，请看上一篇）

```
ps_strategy = tf.distribute.experimental.ParameterServerStrategy()
```

## `tf.distribute.Strategy`在三种API上的使用：Keras、Estimator、Custom Training Loops

### 1. Keras

```
# 这里的Strategy可以换成想用的，因为其他三个还是experimental的状态，建议用这个
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    # 定义模型的时候放到镜像策略空间就行
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    model.compile(loss='mse', optimizer='sgd')
# 手动做个假数据跑一下
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(1000).batch(10)
print('Train:')
model.fit(dataset, epochs=2)
print('\nEval:')
model.evaluate(dataset)
```

### 2. Estimator

```
mirrored_strategy = tf.distribute.MirroredStrategy()
# 在config中加入镜像策略
config = tf.estimator.RunConfig(train_distribute=mirrored_strategy, eval_distribute=mirrored_strategy)
# 把config加到模型里
regressor = tf.estimator.LinearRegressor(
    feature_columns=[tf.feature_column.numeric_column('feats')],
    optimizer='SGD',
    config=config)
def input_fn():
    dataset = tf.data.Dataset.from_tensors(({"feats":[1.]}, [1.]))
    return dataset.repeat(1000).batch(10)
# 正常训练，正常评估
regressor.train(input_fn=input_fn, steps=10)
regressor.evaluate(input_fn=input_fn, steps=10)
```

### 3. Custom Training Loops

```
mirrored_strategy = tf.distribute.MirroredStrategy()
# 在mirrored_strategy空间下
with mirrored_strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    optimizer = tf.train.GradientDescentOptimizer(0.1)
# 在mirrored_strategy空间下
with mirrored_strategy.scope():
    dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(1000).batch(global_batch_size)
    print(dataset)
  # 这里要分发一下数据
    dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
    print(dist_dataset.__dict__['_cloned_datasets'])
def train_step(dist_inputs):
    def step_fn(inputs):
        features, labels = inputs
        logits = model(features)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels)
        loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
        train_op = optimizer.minimize(loss)
        with tf.control_dependencies([train_op]):
            return tf.identity(loss)
  # 返回所有gpu的loss
    per_replica_losses = mirrored_strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
  # reduce loss并返回
    mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    return mean_loss
with mirrored_strategy.scope():
    input_iterator = dist_dataset.make_initializable_iterator()
    iterator_init = input_iterator.initialize()
    var_init = tf.global_variables_initializer()
    loss = train_step(input_iterator.get_next())
    with tf.Session() as sess:
        sess.run([var_init, iterator_init])
        for _ in range(2):
            print(sess.run(loss))
```

[# 分布式训练](http://fyubang.com/tags/分布式训练/) [# 多卡训练](http://fyubang.com/tags/多卡训练/) [# TensorFlow](http://fyubang.com/tags/TensorFlow/)