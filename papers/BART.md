# BART

### 作者

- Facebook团队：Mike Lewis, Yinhan Liu, Naman Goyal

### 摘要

- BART, a denoising autoencoder for pretraining sequence-to-sequence models.
- 训练方式
  - 使用随机噪声函数对文本加入噪声
  - 通过seq2seq模型来重建原始文本
- 最好的方法
  - 随机打乱原始的句子，并且使用新的填充（in-filling）方法
  - in-filling方法：使用单个mask token替代一个text span，强制模型推理被mask句子的长度
- 效果
  - 对于理解任务，和RoBERTa有相当的结果
  - 对于生成任务，有高达6 ROUGE的增益，1.1 BLEU的翻译增益

### 引言

- 改变masked tokens的分布，可以提升自监督训练模型的性能

### 模型

- 目标函数：negative log likelihood of the original document
- 和GPT保持一致，将ReLU改为GeLUs
- 和BERT的不同
  - decoder的每一层都和encoder的最后一层进行cross-attention（和transformer模型一样）
  - BART在最后预测时，没有再加一个前向网络

- 噪声函数（BART的模型结构，可以应用任意的噪声函数来构造噪声文本）
  - token masking：和BERT一样
  - token deletion：模型需要预测哪个位置进行了删除
  - text infilling（Inspired by SpanBERT）：选择一段文本，然后用一个[MASK]进行mask，文本长度取自 $ \lambda = 3$ 的泊松分布。
  - sentence permutation：根据句号，进行打乱顺序
  - document rotation：随机选择一个token，然后将doc进行翻转

### 微调

- 文本分类：decoder的最后一层的最后一个hidden state用于分类
- 序列标注：使用decoder的最后一层的hidden state当成每个词的embedding
- 序列生成：输入进encoder，输出进decoder
- 机器翻译：
  - 将encoder的embedding层替换为一个随机初始化的transformer层，使得新的encoder可以使用和原始BART不同的词表
  - 训练方法：
    1. 只训练新加的transformer层、position embedding、BART第一层的映射矩阵
    2. 训练所有的模型参数

### 对比预训练任务

- base model：6层encoder，6层decoder，hidden size768
- 和BERT进行比较，train 1M步（BERT如何做生成？）
- 预训练任务
  - Language Model（Similar to GPT）
  - Permuted Language Model（Based on XLNet）
    - 选取1/6的tokens，然后自回归地生成他们
  - Masked Language Model（Following BERT）
  - Multitask Masked Language Model（Inspired by UniLM）
    - 随机选择以下几种self-attention的mask：1/6从左到右、1/6从右到左、1/3不mask、1/3前50%的tokens不mask，其余的从左到右mask
  - Masked seq-to-seq（Inspired by MASS）
    - mask一段文本包含50%的tokens

- 训练方法
  - 当成一个标准的seq2seq方法（更适合BART）
  - 将文本当成输出的前缀，使用decoder进行输出（更适合其他模型）
- 任务
  - SQuAD：抽取式问答
  - MNLI：文本对分类
  - ELI5：生成式问答
  - XSum：生成式摘要
  - ConvAI2：生成式问答
  - CNN/DM：生成式摘要
- 结论
  - token masking对于预训练非常重要
  - 从左向右的预训练，可以提升生成任务的性能
  - 双向的encoder对于SQuAD很重要

### 大规模预训练实验

- 实验设置
  - encoder12层，decoder12层，hidden size1024
  - batch size8000，500000步，BPE like GPT
  - 噪声函数：text infilling和sentence permutation。mask 30%的tokens，打乱所有的句子
  - 最后10%的训练步骤，停用了dropout

- 实验结果
  - 判别任务：BART的效果和RoBERTa相当
  - 生成任务：使用了标签平滑的较差熵损失，平滑系数0.1，beam size5
    - 生成式摘要：XSum的ROUGE提升6个点
    - 生成式对话：CONVAI2的自动化指标上，均高于基线
    - 生成式问答：ELI5的ROUGE-L提升1.2%
    - 翻译：高于基线