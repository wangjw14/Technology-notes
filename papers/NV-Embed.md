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













