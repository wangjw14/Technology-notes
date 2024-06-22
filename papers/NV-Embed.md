# NV-Embed

## 摘要

- Decoder only 的 embedding 模型开始超越 BERT or T5 类型的 embedding 模型。
- 本文提出了 NV-Embed 模型，是一个通用的 embedding 模型。
- 模型结构方面：
  - 一个 latent attention layer 来获取 embedding。相比于 <EOS> 或者 mean pooling 效果更好
  - 在 对比学习阶段，去掉了 causal attention mask 
- 模型训练方面：
  - 引入了一个两阶段的对比指令微调方法。
  - 一阶段：在检索数据集上应用对比训练和指令微调。采用了 in-batch 负样本和 精心构造的 hard 负样本
  - 二阶段：将各种非检索数据集混合到指令调整中，从而同时增强检索式和非检索式的效果。
- 效果：
  - 在MTEB上拿到了第一。



## 导言 

- 双向的语言模型在生成 embedding 方面，之前一直是主导的方案。最近一些工作证明 decoder-only LLM可以超过千言的双向 embedding 模型。但是之前的工作，需要使用大量的GPT-4的合成数据，限制了其方法的大量使用。
- E5-Mistral：采用了GPT4的合成数据
- SFR-Embedding-Mistral：混合了检索数据和非检索数据，单个batch内的数据属于同一任务



## 实验

- 采用了 Mistral 7B LLM
- LoRA with rank 16, alpha 32 and dropout rate of 0.1



## 疑问

- loss怎么设计的





# Improving Text Embeddings with Large Language Models

## 摘要

- 仅使用了高质量合成数据， 并且仅需训练1k步
- 不需要复杂的训练流程
- 使用了标准的对比loss进行训练
- 大幅提升了模型效果在MTEB上



## 不足

- 推理成本很高
- 4096维度的embedding存储成本更高
- 只用了手动的prompt



## 前续工作

- BERT- style：Sentence-BERT，SimCSE
- E5、BGE：多阶段的训练范式，先在大量的弱监督文本对上预训练，然后再在高质量的标注数据上微调



- 使用 Mistral-7B，仅仅在合成数据上进行微调，就可以达到和之前在有标签数据上微调的效果。如果将合成数据和有标签数据一起训练，则会大图提升效果（+2%）
- 可以扩展文本输入到32k



- 样本多样性很重要。通过模型生成更多的模板，生成更多的数据。正样本和难负样本都是gpt4生成的。
- 前面告诉模型要做什么任务很重要。我理解是告诉模型进入某一个子空间，在子空间内限制了模型的产出
- 合成数据：数据量大概是1.8亿token，50w数据
- 总训练数据：1.8M数据



# JINA

- CLIP在只有text-only的任务上效果较差
- 本文设计了一个多任务、3阶段的任务来训练，使得text-text和text-image检索效果都很好







































