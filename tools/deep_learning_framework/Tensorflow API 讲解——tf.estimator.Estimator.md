class Estimator(builtins.object)
#介绍
Estimator 类，用来训练和验证 TensorFlow 模型。
Estimator 对象包含了一个模型 model_fn，这个模型给定输入和参数，会返回训练、验证或者预测等所需要的操作节点。
所有的输出（检查点、事件文件等）会写入到 model_dir，或者其子文件夹中。如果 model_dir 为空，则默认为临时目录。
config 参数为 tf.estimator.RunConfig 对象，包含了执行环境的信息。如果没有传递 config，则它会被 Estimator 实例化，使用的是默认配置。
params 包含了超参数。Estimator 只传递超参数，不会检查超参数，因此 params 的结构完全取决于开发者。
Estimator 的所有方法都不能被子类覆盖（它的构造方法强制决定的）。子类应该使用 model_fn 来配置母类，或者增添方法来实现特殊的功能。
Estimator 不支持 Eager Execution（eager execution能够使用Python 的debug工具、数据结构与控制流。并且无需使用placeholder、session，计算结果能够立即得出）。
#类内方法

#####1、__init__(self, model_fn, model_dir=None, config=None, params=None, warm_start_from=None)
构造一个 Estimator 的实例.。
参数：

model_fn: 模型函数。函数的格式如下：
参数：
1、features: 这是 input_fn 返回的第一项（input_fn 是 train, evaluate 和 predict 的参数）。类型应该是单一的 Tensor 或者 dict。
2、labels: 这是 input_fn 返回的第二项。类型应该是单一的 Tensor 或者 dict。如果 mode 为 ModeKeys.PREDICT，则会默认为 labels=None。如果 model_fn 不接受 mode，model_fn 应该仍然可以处理 labels=None。
3、mode: 可选。指定是训练、验证还是测试。参见 ModeKeys。
4、params: 可选，超参数的 dict。 可以从超参数调整中配置 Estimators。
5、config: 可选，配置。如果没有传则为默认值。可以根据 num_ps_replicas 或 model_dir 等配置更新 model_fn。
返回：
EstimatorSpec
model_dir: 保存模型参数、图等的地址，也可以用来将路径中的检查点加载至 estimator 中来继续训练之前保存的模型。如果是 PathLike， 那么路径就固定为它了。如果是 None，那么 config 中的 model_dir 会被使用（如果设置了的话），如果两个都设置了，那么必须相同；如果两个都是 None，则会使用临时目录。
config: 配置类。
params: 超参数的dict，会被传递到 model_fn。keys 是参数的名称，values 是基本 python 类型。
warm_start_from: 可选，字符串，检查点的文件路径，用来指示从哪里开始热启动。或者是 tf.estimator.WarmStartSettings 类来全部配置热启动。如果是字符串路径，则所有的变量都是热启动，并且需要 Tensor 和词汇的名字都没有变。
异常：

RuntimeError： 开启了 eager execution

ValueError：model_fn 的参数与 params 不匹配

ValueError：这个函数被 Estimator 的子类所覆盖
#####2、train(self, input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None)
根据所给数据 input_fn， 对模型进行训练。
参数：

input_fn：一个函数，提供由小 batches 组成的数据， 供训练使用。必须返回以下之一：
1、一个 'tf.data.Dataset'对象：Dataset的输出必须是一个元组 (features, labels)，元组要求如下。
2、一个元组 (features, labels)：features 是一个 Tensor 或者一个字典（特征名为 Tensor），labels 是一个 Tensor 或者一个字典（特征名为 Tensor）。features 和 labels 都被 model_fn 所使用，应该符合 model_fn 输入的要求。

hooks：SessionRunHook 子类实例的列表。用于在训练循环内部执行。

steps：模型训练的步数。如果是 None， 则一直训练，直到input_fn 抛出了超过界限的异常。steps 是递进式进行的。如果执行了两次训练（steps=10），则总共训练了 20 次。如果中途抛出了越界异常，则训练在 20 次之前就会停止。如果你不想递进式进行，请换为设置 max_steps。如果设置了 steps，则 max_steps 必须是 None。

max_steps：模型训练的最大步数。如果为 None，则一直训练，直到input_fn 抛出了超过界限的异常。如果设置了 max_steps， 则 steps 必须是 None。如果中途抛出了越界异常，则训练在 max_steps 次之前就会停止。执行两次 train(steps=100) 意味着 200 次训练；但是，执行两次 train(max_steps=100) 意味着第二次执行不会进行任何训练，因为第一次执行已经做完了所有的 100 次。

saving_listeners：CheckpointSaverListener 对象的列表。用于在保存检查点之前或之后立即执行的回调函数。

返回：
self：为了链接下去。
异常：
ValueError：steps 和 max_steps 都不是 None
ValueError：steps 或 max_steps <= 0
#####3、evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None, name=None)
根据所给数据 input_fn， 对模型进行验证。
对于每一步，执行 input_fn（返回数据的一个 batch）。
一直进行验证，直到：

steps 个 batches 进行完毕，或者
input_fn 抛出了越界异常（OutOfRangeError 或 StopIteration）
参数：

input_fn：一个函数，构造了验证所需的输入数据，必须返回以下之一：
1、一个 'tf.data.Dataset'对象：Dataset的输出必须是一个元组 (features, labels)，元组要求如下。
2、一个元组 (features, labels)：features 是一个 Tensor 或者一个字典（特征名为 Tensor），labels 是一个 Tensor 或者一个字典（特征名为 Tensor）。features 和 labels 都被 model_fn 所使用，应该符合 model_fn 输入的要求。
steps：模型验证的步数。如果是 None， 则一直验证，直到input_fn 抛出了超过界限的异常。
hooks：SessionRunHook 子类实例的列表。用于在验证内部执行。
checkpoint_path： 用于验证的检查点路径。如果是 None， 则使用 model_dir 中最新的检查点。
name：验证的名字。使用者可以针对不同的数据集运行多个验证操作，比如训练集 vs 测试集。不同验证的结果被保存在不同的文件夹中，且分别出现在 tensorboard 中。
返回：
返回一个字典，包括 model_fn 中指定的评价指标、global_step（包含验证进行的全局步数）
异常：
ValueError：如果 step 小于等于0
ValueError：如果 model_dir 指定的模型没有被训练，或者指定的 checkpoint_path 为空。
#####4、predict(self, input_fn, predict_keys=None, hooks=None, checkpoint_path=None, yield_single_examples=True)
对给出的特征进行预测
参数：

input_fn：一个函数，构造特征。预测一直进行下去，直到 input_fn 抛出了越界异常（OutOfRangeError 或 StopIteration）。函数必须返回以下之一：
1、一个 'tf.data.Dataset'对象：Dataset的输出和以下的限制相同。
2、features：一个 Tensor 或者一个字典（特征名为 Tensor）。features 被 model_fn 所使用，应该符合 model_fn 输入的要求。
3、一个元组，其中第一项为 features。
predict_keys：字符串列表，要预测的键值。当 EstimatorSpec.predictions 是一个 dict 时使用。如果使用了 predict_keys， 那么剩下的预测值会从字典中过滤掉。如果是 None，则返回全部。
hooks：SessionRunHook 子类实例的列表。用于在预测内部回调。
checkpoint_path： 用于预测的检查点路径。如果是 None， 则使用 model_dir 中最新的检查点。
yield_single_examples：If False, yield the whole batch as returned by the model_fn instead of decomposing the batch into individual elements. This is useful if model_fn returns some tensors whose first dimension is not equal to the batch size.
返回：
predictions tensors 的值
异常：
ValueError：model_dir 中找不到训练好的模型。
ValueError：预测值的 batch 长度不同，且 yield_single_examples 为 True。
ValueError：predict_keys 和 predictions 之间有冲突。例如，predict_keys 不是 None，但是 EstimatorSpec.predictions 不是一个 dict。
————————————————
版权声明：本文为CSDN博主「HappyRocking」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/HappyRocking/article/details/80500172