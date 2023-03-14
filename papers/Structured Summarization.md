# Structured Summarization

### 作者

Meta AI：Hakan Inan, Rashi Rungta, Yashar Mehdad

### 摘要

- 使用一个端到端的encoder-decoder结构，联合一起，使用生成的方式，训练分段和摘要任务。
- 在分段和摘要任务上都取得了SOTA

### 引言

- 分段情况下的摘要问题，可以看作是一个边界限定的摘要或者分类问题。可以采用“边界限定”的注意力机制。
  - "segmentation-aware" attention mechanism (Zhang et al., 2019; Liu et al., 2021).
- 文献中，摘要问题，并没有和分段问题一起进行探讨。
- 本文的贡献：
  1. 提出了一个判别式分段的baseline
  2. 提出了生成式分段模型，和判别式的分段性能相当
  3. 将分段和摘要一起训练，在分段和摘要结果上都达到SOTA

### 模型

- 判别式分段
  - 使用一个二分类分类器，判断每个句子的BOS token是否是分段的最后一句话

$$
\ell_{\mathrm{seg}}^{\mathrm{CLS}}=\sum_{i=1}^{|S|} \operatorname{BCE}\left(h\left(f\left(t_{i_1}\right)\right), L_{s_i}\right)
$$

- 生成式分段

  - 将代表分段边界的句子index列表，使用分隔符拼接后，作为一个输出的文本序列
    $$
    \ell_{\mathrm{seg}}^{\mathrm{GEN}}=\sum_{i=1}^m \mathrm{CE}\left(\hat{p}_i, 1_{y_i}\right)
    $$

  - 对于第i个句子，使用词表中的第i个token代替原先的BOS token，使得模型对于位置个更加敏感。这个方法显著的提升了模型的性能

- 分段+摘要联合模型
  - 将分段和摘要的loss加在一起



### 实验

- 数据集
  - Wiki-727K：包含727,746英文维基百科文章。train/dev/test：80/10/10%.
  - WikiSection：包含38K英文和德文的维基百科文章，英文文章包括3.6K疾病文档和19.5K城市文档。标签是categorical的。train/dev/test：70/20/10%.
  - QMSum：会议的文本，共232条数据。train/dev/test：68/16/16%.

- 评价指标
  - $P_k$ 
    - 用于估计，随机抽取k个句子，在预测分段和参考分段中，落入不同的分段的概率
    - k设为参考集合分段长度平均值的一半。
  - Rouge
    - 标准的摘要评估指标
    - 将分段的摘要，使用分隔符拼接起来进行计算
  - Label F1
    - 针对WikiSection数据集
- 模型
  - Long T5-base with transient global attention
  - 优化器：AdamW
  - lr：0.0005
  - max_seq_len: 16384 forWiki-727K and WikiSection, 32768 for QMSum
- 验证Structured Summarization的有效性
  - 分段+摘要联合模型的效果是最好的
- 训练trick
  - 先在Wiki-727K上进行预训练，再在其他数据集上进行微调
  - QMSum数据集文本太长，分段点>1000，在预训练集中，没有这么长的分段点。解法：在Wiki-727K的句子前面随机增加空句子，使得文本变长。
  - 输入文本太长，即使每段文本都被截断到200，序列依然很长。解法：每一段按照95进行截断。同时，为了数据增强，分别使用每段20, 50, 200的长度进行截断，构造新的样本。（值得借鉴）





















