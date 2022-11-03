# DeBERTa

### 作者

- Microsoft：Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen

### 摘要

- DeBERTa 
  - Decoding-enhanced BERT with disentangled attention
  - 基于解耦注意力机制的解码增强BERT
- 创新点
  - 解耦注意力机制：内容编码和位置编码分别表示，而不是相加
  - 增强的掩码解码器：在解码层应用绝对位置编码
  - 对抗训练方法
- 效果：
  - DeBERTa-Large（和RoBERTa-Large对比，同时只用了一半的训练数据）
    - MNLI by +0.9% (90.2% vs. 91.1%)
    - SQuAD v2.0 by +2.3% (88.4% vs.90.7%) 
    - RACE by +3.6% (83.2% vs. 86.8%)
  - DeBERTa-1.5 Biliion
    - 单模：在SuperGLUE上首次超越人类表现（89.9 versus 89.8）
    - ensemble：(90.3 versus 89.8)

### 引言

- 解耦的注意力机制

  - deep 和 learning这两个词，当连在一起时，和这两个词出现在不同的句子中时，关系是不一样的
  - 解耦注意力机制（Disentangled attention）

  $$
  \begin{aligned}
  Q_c=H W_{q, c}, K_c &=H W_{k, c}, V_c=H W_{v, c}, Q_r=P W_{q, r}, K_r=P W_{k, r} \\
  
  \end{aligned}
  $$

  $$
  \tilde{A}_{i, j} =\underbrace{Q_i^c K_j^{c \top}}_{\text {(a) content-to-content }}+\underbrace{Q_i^c K_{\delta(i, j)}^r}_{(b) \text { content-oo-position }}+\underbrace{K_j^c Q_{\delta(j, i)}^r}_{\text {(c) position-to-content }}
  $$

  $$
  H_o=\operatorname{softmax}\left(\frac{\tilde{A}}{\sqrt{3 d}}\right) V_c
  $$

  $$
  \delta(i, j)=\left\{\begin{array}{rcl}
  0 & \text { for } & i-j \leqslant-k \\
  2 k-1 & \text { for } & i-j \geqslant k \\
  i-j+k & \text { others. } &
  \end{array}\right.
  $$

  $$
  \delta(i, j)\in[0, 2k)
  $$

  

- 增强的掩码解码器

  - Enhanced Mask Decoder (EMD)
  - 在transformer层后面，在softmax层前面，增加绝对位置编码，用于MLM

- 对抗训练方法

  - Scale-invariant-Fine-Tuning (SiFT)
  - 先将向量归一化，然后在归一化后的向量上加入扰动



### 实验

- 和BERT不同，使用的是BPE词表
- 96 V100 GPUs, 2Kbatch size and 1M steps takes about 20 days
- 全面超越RoBERTa
- 使用RAdam可以达到更好的效果
- 增加相对距离的最大值k，可以提升模型的性能
- Replaced token detection (RTD)的预训练任务，还可以继续提升模型的性能

### 讲解博客

- https://zhuanlan.zhihu.com/p/554273863
- 如何看待微软 DeBERTa、谷歌 T5+Meena 在SuperGLUE 语言基准上超越人类？ - 多头注意力的回答 - 知乎 https://www.zhihu.com/question/438449811/answer/2458546314











