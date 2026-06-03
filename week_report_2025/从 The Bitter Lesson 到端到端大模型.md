# 从 The Bitter Lesson 到端到端大模型



- 模型可以做到多大
  - 第一性原理简陋的估算，可以做到10B
- 如何提升MFU
  - 采用端到端的方式
  - 太多的“精细”设计，使得系统复杂难以维护
- 如何提升模型效果
  - 生成式，not 判别式（判别式是精排模式）
  - code book要合理（id based 太稀疏，10B量级）
  - 解码长度：把推荐行为数据当成一种模态，和语言、图像、声音等对齐

- 后续研究方向
  - RL，如何定义好的推荐结果
  - 推荐Scaling law
  - 行为模态和其他模态对齐，这样才能在文本空间进行推理思考

- 细节
  - 端到端tokenizer：不能做残差的编码，得做类似llm的平行tokenizer
  - item rag



摘要：

- 多阶段系统的问题
  - 计算碎片化、优化不一致性
- onerec是一个端到端生成式方法
  - 计算量提升了10倍，给出了scaling law
  - RL对于端到端系统，有很大的潜力
  - 通过infra优化，提升了MFU。OPEX只有传统系统的10%



导言

- 多阶段系统的问题
  - 碎片化的计算
    - 超过50%的服务资源，用于通讯和存储，而不是高精度计算
    - 精排模型在训练和推理时候的MFU只有4.6%和11.2%。，而LLM的MFU可以到40%
    - 为了实现低延时和高并发，模型往往使用小规模和非计算密集型
  - 目标冲突
    - 多样目标之间的冲突
    - 跨阶段建模的冲突

- OneRec的发现
  - 训练和推理MFU达到了23.7%和28.8%。显著减小了非必要的通讯和存储开销，只需要原始系统的10.6%
  - 将模型的计算量扩大了10倍，以及当模型大小和计算资源扩大时，推荐系统的性能如何被优化
  - RL可以在该框架下有更大的潜力



多模态部分的mask是否双向





架构

- tokenizer

  - 背景
    - item 解空间太大，无法生成id 粒度的token
    - 将 item tokenize 到从粗到细粒度的语义ID，使得相似item之间可以有知识迁移，并且可以更好的泛化到新的item上
    - 除了上下文特征外，还有协同信号。
  - 多模态表征
    - 输入：caption，tag，ASR，OCR，封面，均匀采样的5帧。
    - 采用miniCPM-V-8B，生成了一个1280*512 的 token vectors。然后使用 4层的QFormer 和一个4 X 512 的query，将这些tokens进行了压缩
    - 最终得到4 X 512的表征
  - Item pair
    - user-to-item 检索：将一个user的 positively clicked target item 和一个用户历史上最协同相似的item组成pair
    - item-to-item 检索：通过例如 *the Swing similarity* 这样的相似度分数，组成相似的item pair
  - Loss：
    - 对比学习loss + 生成loss
  - tokenization
    - 采用RQ-Kmeans，一共3层。
    - 比RQ-VAE有更好的重建质量，更好的codebook 使用率

- Encoder

  - 多尺度的特征工程
    - 用户静态通路：uid、性别、年龄。1token
    - 短期通路：最近的20个交互。vid、aid、tag、ts、playtime、dur、label。20 tokens
    - 正反馈通路：vid、aid、tag、ts、playtime、dur、label。256 tokens
    - 终身通路：最大到 100,000 videos。采用2阶段层次化压缩策略
      - 行为压缩，通过层次化Kmeans，聚类数量为 $\lfloor\sqrt[3]{|D|}\rfloor$ ，选择和聚类中心最近的item作为该类的表征
      - 特征聚合，离散特征取离聚类中心最近的item的，连续特征取该类所有item的均值
      - 最长的历史序列为2000，然后通过QFormer，压缩到 128 tokens
  - encoder
    - 总长度405
    - 增加位置编码，双向attention（大号BERT）
  - decoder
    - 每个item前面，增加一个可学习的bos token（why？）
    - 解码器是一个MoE模型，NTP loss

- RL

  - 数据是通过传统的推荐系统得到的。因此无法突破其天花板。通过RL得到更细粒度的偏好
  - 用户偏好对齐
    - 传统方法有很多指标，如点击、喜欢、评论、时长等。通常通过一个加权融合为一个单一指标
    - 通过网络学习一个p-score（preference score）
    - preference model有多个tower，分别学习多个目标作为辅助任务。多个tower的hidden state、用户表征、item表征都输入MLP，得到最终的p-score
    - ECPO，当优势函数 A < 0时，对梯度进行更严格的裁剪，从而保证训练的稳定。同时去除了KL散度

  - format奖励
    - 为保证所有的item被覆盖，语义id的空间要比item空间更大。但是也可能导致非法的item id生成
    - RL的引入，显著的增加了非法item id的生成，尤其是当优势函数 A < 0时
    - 当生成非法的item id时，直接将优势函数设置为0

  - 工业场景对齐
    - 通过将优化目标融合进入奖励系统，可以很方便的使用RL进行优化



- 训练框架
  - 使用了90*8的机器
  - 训练加速：
    - 使用GPU base 的参数服务器，对embedding进行加速
    - ZERO1，数据并行，梯度累积
    - BF16 混合精度训练
    - 编译优化
  - 预训练
    - 数据：每天吞吐18B samples，54B tokens。大约100B samples之后，模型达到收敛
    - 4种不同尺寸的模型
  - 后训练
    - 同时进行RSFT和RL
    - RS：根据播放时长，过滤了50%播放时长少的 sessions



- 评估
  - metrics：loss、p- score、xtr
  - Scaling
    - parameters scaling：模型越大，loss越低
    - Feature Scaling：输入特征越多，效果越好
    - Codebook Scaling：从8K扩大到32K，效果提升，特别是播放时长相关指标
    - Infer Scaling：生成的items越多，效果越好。但是超过512后，收益开始变小

















$l$ 层transformer模型的可训练模型参数量为 $l\left(12 h^2+13 h\right)+V h$ 。当隐藏维度 $h$ 较大时，可以忽略一次项，模型参数量近似为 $12 l h^2$ 。

对于一个 $l$ 层的transformer模型，输入数据形状为 $[b, s]$ 的情况下，一次训练迭代的计算量为 $l *\left(24 b s h^2+4 b s^2 h\right)+2 b s h V$ 。当隐藏维度 $h$ 比较大，且远大于序列长度 $s$ 时，我们可以忽略一次项，计算量可以近似为 $24 b s h^2 * l$ 。

输入的tokens数为 $b s$ ，存在等式 $\frac{24 b s h^2 l}{12 h^2l \times b s}=2$ 。我们可以近似认为：在一次前向传递中，对于每个token，每个模型参数，需要进行 2 次浮点数运算，即一次乘法法运算和一次加法运算。



如果做一个和LLM同架构的模型，计算总量为 $2 N D$， $D=bs$ 是输入输出token总量 ， $N=12 l h^2$是模型的尺寸。





真正有效的路径都是那些能承载更大算力投入的方案。推荐过去的迭代趋势也满足这一特性，从规则到协同矩阵分解/逻辑回归，再到深度学习以及进一步的序列建模。推荐系统的算力投入每一代增长几个数量级，效果也对应大幅提升。

当前的推荐模型架构，广义scaling law最明显的仅在：1. 行为序列长度 2. 打分候选集。

一年前，我手推荐系统中，计算最密集的一个精排模型在A10上训练和serving MFU都只有个位数，而 LLM 在训练时可以在H100上把MFU做到40-50%（不考虑fp8）。这个差距主要在于精排模型里特殊的设计算子太多了，做了非常多算力有限假设下“精细”设计，这些“精细”设计都是短期有效，长期会变成负担的工作，所以推荐模型的MFU越来越低。















https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf

分析transformer模型的参数量、计算量、中间激活、KV cache - 回旋托马斯x的文章 - 知乎

https://zhuanlan.zhihu.com/p/624740065