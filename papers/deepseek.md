# deepseek

长期主义



## DeepSeek LLM

4次长期主义

### 摘要

- scaling law有多种结论
- 长期主义视角

- DeepSeek LLM 超越 LLaMA-2 70B

### 引言

- 开源社区大都聚焦于训练一个特定大小的模型，而对于最重要的scaling law没有做深入研究。
- 随着预算的增长，如何扩大模型和数据的大小，之前的文献得出了不同的结论。
- 超参数如何选择也没有被很好的讨论。

- Scaling laws of hyperparameter：随着模型小的扩大，该如何选择超参数（batch size和learning rate）。
- Scaling laws of data/model：模型和数据该如何扩大，才能得到最好的效果。
- 使用不同的数据集的时候，scaling laws会有不同的结论，表明数据质量非常重要。
- 使用了llama的架构，2T tokens，将cosine lr改为multi-step lr

- DeepSeek LLM 超越 LLaMA-2 70B，超越GPT-3.5

### 预训练

- 数据：
  - 数据处理：去重、过滤、混合
  - 去重时在最大的数据范围内进行去重，可以去掉更多的重复内容
  - 混合时，增加长尾数据的占比，得到更加平衡和全面的数据集
  - tokenizer：Byte-level Byte-Pair Encoding (BBPE)

- 模型结构
  - 和llama保持一致，引入了GQA

- 超参数
  - 使用了multi-step lr，2000 steps warmup，当训练了80%的tokens 之后，降低到 31.6% * lr。当训练了90% tokens 之后，降低到 10% * lr。
  - 使用了multi-step lr，允许重复使用第一阶段的训练，为持续训练提供了独特的便利

### Scaling Laws

- 我们建立了超参数的缩放定律，为确定最佳超参数提供了一个经验框架。
  - 模型越大，batch越大
  - 模型越大，lr 越小
- 我们采用非嵌入的每token浮点运算次数（FLOPs/token）M 来代替模型参数 N 以表示模型规模，这会带来更准确的最优模型 / 数据放大分配策略，并能更好地预测大规模模型的泛化损失。
  - 通过我们的方法，可以精确预测给定预算情况下，模型最终的收敛情况
- 预训练数据的质量会影响最优的模型 / 数据放大分配策略。数据质量越高，增加的计算预算就应该更多地分配给模型缩放。
  - 数据质量越高，模型越大。





### 结论

- 提出了最优的 model/data 扩展策略。

- 接近最优的超参数选择策略。
- 数据质量是造成scaling law结论不一致的主要根源。
- 避免了测试结果美化和隐瞒一些dark secrets。



## Deepseek MoE

### 摘要

- MoE是一种可以降低计算成本的有效架构
- DeepSeek MoE：将专家划分为更细粒度，保持有一些共享专家
- 可以用更少的计算量达到更好的效果
- 可以scaling到145B，和DeepSeek 67B效果差不多，计算量只有28.5%



### 引言

- 主要贡献：
  - 模型结构创新：细粒度的专家划分和共享专家
  - 仔细验证了Moe结构的效果，2B模型可以达到接近理论的上限
  - 并且可以scaling 145B





## Deepseek V2

### 摘要

- deepseek v2：训练经济，推理高效
- 结构：MLA、DeepSeek MoE
- 训练成本降低42.5%，推理速度增加5.76倍，KV cache节省93.9%



### 结构



Dense 模型的计算量分布

|      | Attention | MLP    |
| ---- | --------- | ------ |
| 公式 |           |        |
| 举例 | 2.7e12    | 4.4e12 |

$$
8bsh^2+4bs^2h
$$

$$
16bsh^2
$$

$$
b=1,s=4096,h=8192
$$



Dense 模型的显存分布









MLA

MoE



问题：负载不均衡

- 专家级别的平衡loss
- 设备级别的平衡loss
- 通讯loss
- token 丢弃策略











## Deepseek V3





## Deepseek R1





模型结构：

效果和低成本，我都要



模型和架构协同设计



其他：

专注（不做商业化、不做dense model）

要做开源，

团队建设

不融资





|                                         | DeepSeek LLM | DeepSeek V2   | DeepSeek V3   |
| --------------------------------------- | ------------ | ------------- | ------------- |
| Total Params                            | 67B          | 236B          | 671B          |
| Activated Params                        | 67B          | 21B           | 37B           |
| Total Experts                           | -            | 2+160 experts | 1+256 experts |
| Activated Experts                       | -            | 2+6 experts   | 1+8 experts   |
| Tokens                                  | 2T           | 8.1T          | 14.8T         |
| MMLU                                    | 71.3         | 78.4          | 87.1          |
| 训练成本（H800 GPU  hours/每1T tokens） | 300K         | 172.8K        | 180K          |
| 生成速度（tokens/sec）                  | 8.7k         | 50k           |               |







## 参考文献

https://martinfowler.com/articles/deepseek-papers.html

【Deepseek 系列】V1/V2 /V3 /R1 技术演进之路速读 - 乞力马扎罗不说话的文章 - 知乎
https://zhuanlan.zhihu.com/p/21752444414

deepseek技术解读(3)-MoE的演进之路 - 姜富春的文章 - 知乎
https://zhuanlan.zhihu.com/p/18565423596