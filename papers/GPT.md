# GPT

## GPT3的应用

- Hacker News上用GPT3生成的博客文章 https://adolos.substack.com/p/feeling-unproductive-maybe-you-should
- MIT Technology Review，一个大学生用AI生成的博客，糊弄了上万人 https://www.technologyreview.com/2020/08/14/1006780/ai-gpt-3-fake-blog-reached-top-of-hacker-news/
- GPT-3: Demos, Use-cases, Implications, https://towardsdatascience.com/gpt-3-demos-use-cases-implications-77f86e540dc1

- https://gpt3demo.com/
- GitHub Copilot：根据注释写代码

## 历史

- Transformer：2017.06，引用29k
- GPT：2018.06，引用3k
- BERT：2018.10，引用27k
- GPT-2：2019.02，引用5k
  - 更大的数据集，更大的模型，比BERT large更大。非常适合做 zero shot，效果没有那么惊艳。
- GPT-3：2020.05，引用3k
  - 比GPT2，数据和模型都大了100倍，暴力出奇迹



## GPT

- Improving Language Understanding by Generative Pre- Training
- 作者
  - Alec Radford
    - DCGAN：使用卷积神经网络替代GAN里面的MLP
    - PPO：强化学习中的一个常见优化算法
    - GPT三部曲
  - Ilya Sutskever
    - AlexNet的二作
    - OpenAI的CTO

- 摘要

  - 在没有标号的文本上做预训练，是迈出了一大步
  - 做zero-shot，是走出来另外一大步
  - 预训练之后，再微调，这种模式是一种大的创新
  - 9/12个任务上，效果达到最好

- 导言

  - 使用无标号文本的2个难点：
    - 使用没有标号的文本时，如何选择优化目标和损失函数（语言模型？机器翻译？文本一致性？）
    - 怎么样把学到的文本表示，传递到下游的子任务上

- Framework

  - 如何在没有标号的数据集上预训练
    $$
    L_1(\mathcal{U})=\sum_i \log P\left(u_i \mid u_{i-k}, \ldots, u_{i-1} ; \Theta\right)
    $$

  - 怎么样做微调
    $$
    L_2(\mathcal{C})=\sum_{(x, y)} \log P\left(y \mid x^1, \ldots, x^m\right)\\
    L_3(\mathcal{C})=L_2(\mathcal{C})+\lambda * L_1(\mathcal{C})
    $$

  - 怎么样对子任务表示文本的输入

- 和BERT的对比

  - BERT base：L=12, H=768, A=12, Para=110M
  - BERT large：L=24, H=1024, A=16, Para=340M，模型是3倍，数据是4倍
  - BERT的数据集：BooksCorpus 800M words，新增English Wikipedia 2500M words，总数是是GPT的4倍
  - GPT的总体精度：75.1，BERT base：79.6，BERT large：82.1



## GPT-2

- Language Models are Unsupervised Multitask Learners
- 摘要
  - WebText，百万级别的文本
  - zero-shot
- 方法
  - 采用了prompt的方式
  - 使用了Reddit中至少3个karma值的数据，一共45million的链接，8million 文档，40GB的文本
  - 没有使用CommonCrawl，信噪比太低，清理很麻烦
- GPT-2的参数：L=48, H=1600, Para=1542M



## GPT-3

- Language Models are Few-Shot Learners
- 摘要
  - 特别大：GPT-3特别大，175B，是之前的非稀疏模型的10倍大
  - Few shot的时候，不需要计算梯度和微调
  - 效果特别好：可以生成人类难以分辨是否是机器生成的文章

- 导言
  - In context learning
  - 评估方法：
    - few shot learning
    - one shot learning
    - zero shot learning

- 模型
  - GPT-3的参数：L=96, H=12288, Para=175B
  - 使用了sparse transformer
  - 模型变大的时候，计算复杂度和宽度成平方关系，跟层数是线性关系
  - 模型变大的时候，batch size是增大的（3.2M），lr是降低的

- 数据
  - 清理CommonCrawl
    - 用LR做二分类，去判断质量
    - 去重，LSH算法（信息检索的一个常用技术）
    - 加了一些已知的高质量数据集
  - 训练的一个batch中，对于不同来源的数据采样做了不同的权重。保证有一定比例的高质量数据
- 训练
  - 使用的是DGX-1的集群，有很高的带宽

- 结果
  - 针对一个模型结构和训练数据，如果想线性的降低loss，计算量需要指数级增加
- 局限性
  - 对于生成长文本比较难
  - 模型的局限性，只能从前向后预测
  - 无法判断单个token的重要性
  - 样本有效性不够









