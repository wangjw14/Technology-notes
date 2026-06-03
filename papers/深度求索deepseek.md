# 深度求索DeepSeek背后的底层逻辑

大家好，我是JOYWIN。DeepSeek目前已经火遍全球，网络上对他的解读也非常多。但我仍然好奇一些问题:

- DeepSeek不同版本是如何一次次进化，直至效果逆天的？
- 为什么众多大模型玩家中，脱颖而出的是他？对于我们又有哪些借鉴意义？
- DeepSeek的成功，为未来大模型的进化提供了哪些方向？

带着这样的疑问，我通读了 DeepSeek 几篇主要的技术报告，试图梳理出 DeepSeek 每一次进化的详细历程，以及隐藏在每一次进化背后的底层逻辑。当我梳理完这一切，我只能感叹，DeepSeek是真的强（比想象的更强）！本文将先回顾整个DeepSeek的技术报告历史，总结每一版本的主要贡献，然后梳理总结他们的底层逻辑。先放出本文的思维导图。

![ds_mindmap](/Users/joywin/joywin/Technology-notes/papers/pics/ds_mindmap.png)

OK，下面就开始一起深度求索DeepSeek（Deep Seek the DeepSeek）！

## 1. DeepSeek LLM/MoE: 长期主义视角下，最优的大模型的组成要素

大模型的训练成本非常之高，尤其是在预训练阶段。而影响模型训练的因素又非常多，比如模型/数据的计算预算分配、数据质量判断、超参数设置、模型结构（dense vs MoE）的选择等等。如果最初从实验设计上没有选到正确的技术路线，无疑会造成资源的巨大浪费。因此，DeepSeek在最开始的时候，就从长期主义的视角，专注于研究影响模型效果的主要因素，从而确认了通往最优大模型的技术路径，为后来的发展奠定了扎实的基础。

本章主要涉及2024年1月的2篇文献，[DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/pdf/2401.02954) 和 [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)。

### 1.1 Scaling Laws

Scaling Laws 是指，当其他因素不成为瓶颈时，计算量、数据集大小、模型参数量这3个因素中的单个因素指数增加时，模型的 loss 会线性的下降。Scaling Laws 为计算资源如何分配给模型和参数提供了重要指导。早期 OpenAI 和 Google 分别针对 Scaling Laws 进行过研究。然而，他们分别的出的结论却不尽一致。从 chatGPT 之后，开源社区就不断涌现出不少优秀的模型，如 Llama 系列等。然而，对于这么重要的 Scaling Laws ，进一步的研究却很少。因此，DeepSeek 先对 Scaling Laws 进行了探究，并得出了自己的 Scaling Laws。

#### 1.1.1 Model/Data Scaling Laws

Scaling Laws 首先要回答的问题就是，给定固定的计算预算，如何分配模型参数大小和数据量。假设计算预算为 $C$，模型参数量大小为 $N$，数据规模大小为 $D$。则根据之前 Scaling Laws 的研究，计算预算和模型/数据规模之间的关系可以近似表示为 $C=6ND$。同时，之前的 Scaling Laws 中，对于模型参数量的定义并不一致，分别是非 embedding 参数量 $N_1$ 和全部参数量 $N_2$，对应的模型参数量也可以对应表示为 $6N_1$ 或者 $6N_2$。然而，由于 $N_1$ 和 $N_2$ 都没有考虑注意力操作的计算开销，而 $N_2$ 还包含了对模型容量贡献较小的词汇表部分，因此，它们都存在显著的近似误差。因此，DeepSeek 去除了词汇计算部分的参数量，然后将注意力的计算开销也包含在内，重新定义了模型参数量：non-embedding FLOPs/token，将其表示为 $M$。不同的模型参数规模的计算方式分别如下所示。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/model_capacity.png" alt="model_capacity" style="zoom:25%;" />

在用 $M$ 表示模型规模之后，训练目标可以更清晰地描述为：给定计算预算 $C=MD$，找到最优的模型规模 $M_{\text{opt}}$ 和数据规模 $D_{\text{opt}}$，以最小化模型的泛化误差。经过大量实验之后，DeepSeek拟合出了总计算量、模型规模和数据大小之间的关系，如下所示。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/model_data_sl.png" alt="model_data_sl" style="zoom:25%;" />

利用这样的计算关系，DeepSeek准确预测了最大模型的训练效果。如下图所示。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/model_data_sl2.png" alt="model_data_sl2" style="zoom:30%;" />

图中虚线表示对较小模型（灰色圆圈）进行 Scaling Laws 拟合的结果。蓝色星形标记代表 DeepSeek LLM 的 7B 和 67B 版本。可以看到，在计算 $10^{18}-10^{20} FLOPs$ 之后，就可以精确预测模型在 $10^{24} FLOPs$ 时的效果，足足节省了了4-6个数量级的计算。

#### 1.1.2 Parameters Scaling Laws

除了模型规模和数据大小，超参数的选择也会影响模型的性能。而这一点，之前的研究很少。同样，通过大量的实验，DeepSeek 也拟合出了最优的 batch size、learning rate与计算量之间的关系，具体如下：

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/param_sl.png" alt="param_sl" style="zoom:25%;" />

根据拟合关系，同样可以精准预测训练更大模型所需要的最优 batch size 和 learning rate。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/param_sl2.png" alt="param_sl2" style="zoom:30%;" />

上图中灰色圆圈表示泛化误差超出最小值不超过 0.25% 的模型。虚线表示对较小模型拟合的 Scaling Laws。蓝色星形标记代表 DeepSeek LLM 的 7B 和 67B 版本。可以看到，最优超参数的预测也非常准确。

此外，从图中也可以看到，随着计算预算的增加，最优的 batch size 是逐步增大的，而最优的 learning rate 是逐步降低的。这也和日常训练模型的直觉是吻合的。

#### 1.1.3 Data Quality Scaling Laws

在模型训练中，数据质量无疑是非常关键的。这方面的 Scaling Laws 研究也很少。DeepSeek 在不断构建新的更高质量的数据集的过程中发现，不同数据质量的数据，也会影响 Scaling Laws。具体而言，随着数据质量的提高，模型扩展指数逐渐增加，而数据扩展指数则逐渐减少。也就是说，在数据质量较高的情况下，增加的计算预算应更多地分配给模型而非数据。这一发现或许也能解释早期关于 Scaling Laws 研究中观察到的最优模型/数据扩展分配策略的显著差异。

这一结论，和直觉也是吻合的，也就是说，数据质量越高，模型训练的就会越好。训练得到同样好的模型，需要的数据数量就会下降。更进一步，DeepSeek通过观察训练过程中，loss的下降情况，就可以发现数据质量是否存在问题。

综合三条Scaling Laws，DeepSeek无疑是把炼丹技术提升了炉火纯青的地步。在训练过程中，如果已经采用了最优的配置和超参数，可以直接排除掉超参数不合理的原因，集中精力排查其他方面的问题。为后续的实验节省了大量的试错时间。

### 1.2 Dense vs MoE

除 Scaling laws 之外，在大模型训练中，还有一个重要选择，就是模型结构选择 dense 还是 MoE 结构。Dense结构是开源大模型中采用较多的结构（比如最有名的Llama系列），但同时很多研究显示 MoE 也是一种很好的选择。比如，2022年，Google 在 GLaM 论文中就提出，使用 MoE 结构可以在降低成本的同时，实现比 dense 结构更好的效果，如下图所示。图中，采用 MoE 结构的 GLaM 可以在成本上下降一半左右，但是可以在精确率方面有提升。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/glam.png" alt="glam" style="zoom:40%;" />

究竟是跟随 Llama 的步伐训练 dense 结构，还是选择 GLaM 的 MoE 路线？DeepSeek决定自己动手实验。DeepSeek 先实现了较小规模的 MoE 模型（2B和16B）。同时，针对传统的MoE模型，做了2点改进，分别是：1）将MoE结构中的专家粒度划分的更细，使得每个专家之间的冗余度更低；2）增加了一些共享专家，学习一共公共的知识和能力。下表对比了 DeepSeekMoE 16B模型，和 LLaMA2 7B的结果。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/moevsdense.png" alt="moevsdense" style="zoom:30%;" />

可以发现，只使用了 LLaMA2 7B 成本的 40%，DeepSeekMoE 16B 就在大部分任务上超过了 LLaMA2 7B，验证了之前 Google 结论的正确性。之后，DeepSeek 通过进一步的实验，验证了 MoE结构可以将规模扩大到 145B 大小，同时保持稳定的训练。那么，结论就显而易见了。既然 MoE 结构可以用更低的成本得到更好的模型，同时还可以 scaling 到很大的规模，同时保持训练稳定，那么毫无疑问，MoE结构才是训练最优大模型的最佳选项。

### 1.3 总结和推论

可以看到，在2024年初，DeepSeek 就以长期主义的视角，对影响最优大模型性能的因素进行了系统而全面的探索，为后续的发力打下了坚实的基础。同时，站在2025年的时间点，看到DeepSeek的后续动作之后，可以得到一个重要推论：

**推论一：从2024年1月，DeepSeek就确立了All In MoE的技术路线，彻底放弃了dense模型。**

从后续发布的模型来看，DeepSeek V2、DeepSeek V3、DeepSeek VL2、DeepSeek Coder V2，全部采用了 MoE 结构，再也没有继续发布 dense 系列的主模型。作为对比，LLama系列全系列是 dense 模型；Qwen系列以 dense 为主，只有少部分参数量较小的 MoE 模型，没有特别大的 MoE 发布；Mixtral 最初发布了一些 MoE 模型，但后续也开始发布 dense 模型，没有继续发布 MoE 模型。

后文我们会看到 MoE 结构为后续 DeepSeek 在效果和低成本方面的突破打下了重要的基石，而在很早的时候，就能有这样的笃定和勇气，让我非常佩服他们的果敢。这很容易让人想到早期的 OpenAI，当初所有人都以为双向注意力的 BERT 结构优于单向注意力的 GPT 结构，只有OpenAI笃定地选择了 GPT 路线，直到后来 GPT3 震惊了所有人。DeepSeek 也在一年后，用极致的成本和效果的 Mo E模型，震惊了全世界。



## 2. V2/V3：效果和成本，既要又要

### 2.1 大模型的效果和成本

大模型训练和推理，都需要很高的成本。然而，如果想要将大模型应用于实际的场景，成本依然是一个必须要考虑的因素。如果同时考虑效果和成本2个因素，可以将不同的场景分为四个象限。如下图所示。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/sixiangxian.png" alt="sixiangxian" style="zoom:50%;" />

当效果和成本不能同时满足时，“效果优成本低”和“成本低效果差”的模型分别还是有一些应用场景的，但是最佳选择依然是“效果优且成本低”的模型。当大部分人在追求靠近第一象限的时候，DeepSeek 把目标确定在了第二象限。而当第二象限的“最佳选择”模型出现的时候，他就可以做到一统江湖，淘汰其他的竞争对手。

在看 DeepSeek 如何同时优化模型的效果和成本之前，我们先看看传统的 transformer 结构的特点。从传统结构的特点出发，也就可以发现优化的思路和出发点。首先是**计算量分布**。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/transformer_compute.png" alt="transformer_compute" style="zoom:50%;" />

如上图所示，一个传统的 transformer block在前向传播中，主要的计算量分别是 attention模块和 FFN 模块。其中，attention模块的计算量为 $8bsh^2+4bs^2h$，FFN 模块的计算量为 $16bsh^2$ 。其中，batch size 为 $b$，序列长度为 $s$，hidden state为 $h$。以 DeepSeek 67B 在预训练时候的配置为例，取$s=4096,h=8192$，可以看到，FFN的计算量占总计算量的62%。因此如果想要降低模型整体的计算量，最先应该优化的就是 FFN 部分的计算量。

其次，在看推理时候的**显存分布**。推理阶段主要存储消耗是两部分： 模型参数和 KV Cache。根据参考文献的推算，以Qwen-72B模型为例，模型参数需要的显存为144G（每个参数占2Bytes），而KV Cache需要占据的显存数量为 $4blhs$。当同样以Qwen 72B为例时，模型层数 $l=80$，hidden state 维度 $h=8192$，batch size $b=32$，序列长度 $s=4096$ 时，计算可得，KV Cache一共需要占据显存 343 GB，显存消耗是模型参数的2倍还要多。所以，KV Cache的优化非常重要。当通过优化降低 KV Cache 的显存要求之后，推理可以采用更大的 batch size，可以提升推理速度，同时，更大的batch size，也可以更充分的利用 GPU 的并行计算的能力，进一步加速推理的效率。

|              | 模型参数显存 | KV Cache显存 |
| ------------ | ------------ | ------------ |
| b=1, s=2048  | 144GB        | 5.36GB       |
| b=32, s=4096 | 144GB        | 343 GB       |

因此，通过上述分析可以看到，训练时成本的最大部分在 FFN 模块，推理时影响性能最大的部分是 KV Cache。因此，这也是想要降低训练和推理的成本，最应该优化的就是这两部分，下文可以看到，DeepSeek 是如何针对这两部分进行优化的。

本章同样涉及2篇文献，[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/pdf/2405.04434) 和 [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)。

### 2.2 DeepSeekMoE

#### 2.2.1 细粒度专家划分和共享专家

MoE 结构通过将 FFN 模块稀疏化，可以大幅降低模型的计算量。然而，传统的MoE模型，还面临专家之间参数冗余等问题。为了能够降低不同专家之间的冗余度，从而更进一步降低成本，DeepSeek也对MoE结构进行了优化，主要是细粒度专家划分和共享专家。

下表列举了一些MoE模型的主要参数，比如 Mixtral 8x7B、Google 的 GLaM 等。

| 模型              | 总参数 | 激活参数 | 激活参数占比 | 专家总数 | 激活专家数 | 激活专家占比 | 专家维度 |
| ----------------- | ------ | -------- | ------------ | -------- | ---------- | ------------ | -------- |
| Mixtral 8x7B      | 47B    | 13B      | 27.7%        | 8        | 2          | 25%          | 14336    |
| GLaM (64B/64E)    | 1.2T   | 96.6B    | 8%           | 64       | 2          | 3.1%         | 32768    |
| DeepSeek MoE 145B | 145B   | 22B      | 15%          | 4+128    | 4+12       | 12%          | 2048     |
| DeepSeek V2       | 236B   | 21B      | 8.9%         | 2+160    | 2+6        | 4.8%         | 1536     |
| DeepSeek V3       | 671B   | 37B      | 5.5%         | 1+256    | 1+8        | 3.5%         | 2048     |

通过上表可以看到，和之前的 MoE 模型相比，DeepSeek 将 MoE 结构中的专家粒度划分的更细，**专家维度**和之前相比下降了一个数量级，同时**专家数量**上升了一个数量级。更细粒度的专家可以使各个专家更好的拟合部分适合自己的场景，减少冗余度。同时，更多的专家，可以获得更多的不同专家的组合。这样可以不同的专家可以组成更多样的组合，从而适应更多的场景。

同时，DeepSeekMoE还设置了一些共享的专家（专家总数中的1+256，代表1个共享专家+256个路由专家），这些共享的专家会一直保持激活状态，保证一些通用的知识的学习。

此外，对比DeepSeek不同版本模型的参数变化，也可以看到，随着版本迭代，模型的激活参的占比、激活专家占比、共享专家数都在**不断下降**，从而进一步压缩训练和推理的成本。而通过扩大模型的整体参数、更好的训练方式，进一步提升了模型的整体效果。

根据DeepSeek V2的统计，采用 MoE 结构之后，模型成本得到了巨大的下降，相比 DeepSeek LLM 67B，**训练成本降低了 42.5%**，同时还保持了**效果提升**，可以说是**真的做到了既要又要**。

#### 2.2.2 如何做好专家路由

既然MoE结构既能提高效果，同时还能降低成本，为什么别的厂商不选这一路线呢？那就是因为 MoE 结构的训练会非常不稳定。在模型训练中，往往容易发生某些专家过载，其他模型几乎没有被激活的负载不均衡问题。

DeepSeek 也知道 MoE 结构更难训练，但既然目标是效果和成本要同时优化，**All In MoE**，那就必须逼迫对模型训练进行创新，从而更够更好也更稳定的训练 MoE 模型。 下表列举了不同版本中，DeepSeek尝试的各种策略。

|              | 专家级均衡loss | 设备级均衡loss | 通讯均衡loss | 设备受限路由 | 无辅助损失 | sequence均衡loss |
| ------------ | -------------- | -------------- | ------------ | ------------ | ---------- | ---------------- |
| DeepSeek MoE | ✅              | ✅              |              |              |            |                  |
| DeepSeek V2  | ✅              | ✅              | ✅            | ✅            |            |                  |
| DeepSeek V3  |                |                |              | ✅            | ✅          | ✅                |

可以看到，DeepSeek 做了不同的尝试，从最初的辅助均衡损失函数（专家级、设备级、通讯级别），到后来为了防止负载均衡的 loss 太大，从而影响模型的主 loss（next token prediction）的优化，进一步提出了无辅助损失策略。经过不断的迭代，使的模型可以训练稳定，同时效果更优。

无辅助均衡的路由策略如下：

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/auxiliary_loss_free.png" alt="auxiliary_loss_free" style="zoom:50%;" />

其中，门控值 $s_{i,t}$ 仍然采用原始的 token 到专家的匹配值，偏置项 $b_i$ 仅用于路由。在训练过程中，会对整个批次的专家负载情况进行监控。如果某个专家负载过重，则会减小对应的偏置项；反之，如果某个专家负载不足，则会增加对应的偏置项。通过这种动态调整，DeepSeek-V3 在训练期间保持了**专家负载平衡**，并且相比仅通过纯辅助损失来鼓励负载均衡的模型，表现出了**更好的性能**。可以说**又一次做到了既要又要**。

### 2.3 MLA

根据2.1，我们已经知道，在推理时，KV Cache 会占据很大一部分显存。因此，对于比较大的模型，优化KV Cache就是必须要进行的操作。之前一般有 MQA、GQA 进行优化，但是这两种操作往往会带来效果上的下降，因此，DeepSeek提出了自己设计的MLA方法。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/mla.png" alt="mla" style="zoom:50%;" />

在 MLA 中，不需要缓存所有的 KV，而是对 KV 压缩到了潜语义空间，显存中只需要缓存压缩后的潜向量即可，大大降低了对于显存的要求。同时，在计算中，可以通过矩阵吸收的方法，进行计算的简化。

为了更直观对比，下表中给出了以Qwen-72B参数设置为基础，$b=32, s=4096$ 时，采用不同方法，实际的KV Cache占用量。

| 方法 | KV Cache per token | 72B 模型实际占用 |
| ---- | ------------------ | ---------------- |
| MHA  | $2n_hd_hl$         | 343 GB           |
| GQA  | $2n_gd_hl$         | 42.9GB           |
| MQA  | $2d_hl$            | 5.36GB           |
| MLA  | $4.5d_hl$          | 12GB             |

可以看到，通过使用 MLA，将 KV Cache 的显存占用量，**降到了MHA的3.5%**，相当于$n_g=2.25$ 时的GQA。同时通过消融实验，可以看到，采用 MLA 的效果比 MHA **效果更好**。 **既要又要**成就再一次达成。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/mlavsmha.png" alt="mlavsmha" style="zoom:30%;" />



### 2.4 FP8 混合精度训练

为了加速模型训练，DeepSeek V3中，采用了fp8的混合精度计算框架。具体来说，大多数核心计算内核，即 GEMM（General Matrix Multiply）操作，以 FP8 精度实现。而对精度要求更高的Embedding模块、输出头、MoE 门控模块、归一化操作符和注意力操作符，则采用 BF16 或 FP32的高精度进行计算。

我们仔细看下 fp8 是如何进行量化计算的。首先，假设一个维度为 $h$ 的行向量和列向量进行相乘，一共需要做h次乘法和h-1次加法。其中，乘法一般不会触发 fp8 精度的溢出，但是如果 $h$ 足够大，由于累加的次数足够多，常常会导致溢出。因此，fp8 计算的原理，首先就是要对向量或者矩阵进行分块，避免原始大维度的矩阵直接进行相乘。一般来说，分块有以下几种分法：

**per tensor**：对一个tensor（二维）量化成低比特，并用一个scale表示。

**per token**：对一行/一列元素量化成低比特，并每一列都用一个scale表示。

**group wise量化**：对特定个元素为一组，每组元素用一个scale进行表示。

**tile wise**：对特定的一块区域进行量化，并对这块特定的位置取一个scale（比如128x128）

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/fp8_quant.png" alt="fp8_quant" style="zoom:60%;" />

在DeepSeek V3中，激活值 activation 采用的是 group wise 的分块量化，而权重 weight 采用的是 tile wise 量化。通过将大的矩阵分为较小的块（比如128维），那么较少的累加次数，就能大大降低溢出的风险。当分块的 fp8 计算完成之后，再将各个分块的值，再用 fp32进行一次累加，就可以在保证不溢出的情况下，得到最终的计算结果。具体的流程图如下所示。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/fp8_compute.png" alt="fp8_compute" style="zoom:45%;" />

有了上面的基础，对于论文中的图就可以更容易的理解。DeepSeek V3中一共有3个部分，采用了fp8精度的混合计算，分别是前向传播，对权重求导和对输入求导。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/fp8train.png" alt="fp8train" style="zoom:45%;" />

这种设计理论上将基于 BF16 方法的**计算速度提高一倍**。此外，FP8 Wgrad GEMM 允许激活在反向传播中以 FP8 格式存储，这**显著减少了内存消耗**。但同时，FP8 训练的**相对 loss 误差持续低于 0.25%**。

### 2.5 DualPipe

在DeepSeek V3 中，不同的专家是通过专家并行，分布在不同的 GPU 上的。也就是说，前向传播时，attention层的计算结果，需要通过一次 all-to-all 通讯，将计算结果分别传输给需要计算的专家，当专家层的 MLP 层计算完成之后，还需要一次 all-to-all 的通讯，将计算结果再汇集到一起给到下一层的 attention。也就是说，一次计算过程一共包含：attention, all-to-all dispatch, MLP,  all-to-all combine 四个组件。

可以看到，这种专家并行的方式，引入了大量的通讯，如果处理不恰当，会造成大量的阻塞，严重影响模型的性能。为了解决这一问题，DeepSeek V3中引入Dual Pipeline。DualPipe的核心就是让计算和通信相互重叠，提高整体训练效率，降低因通信造成的等待时间（即减少流水线气泡）；DualPipe让模型在进行计算的同时，后台已经开始准备下一步需要的数据传输。这种设计确保了通信开销被很大程度地隐藏在计算过程中，极大提升了整体效率。

**前向传播过程**：

![DualPipe_forward](/Users/joywin/joywin/Technology-notes/papers/pics/DualPipe_forward.png)

**反向传播过程**：

![DualPipe_backward](/Users/joywin/joywin/Technology-notes/papers/pics/DualPipe_backward.png)

值得一提的是，DualPipe 是一种**专门针对大规模 MoE 模型设计**的流水线并行。为了能够隐藏 attention 和 MLP 中间的 all-to-all  通讯，才专门设置了双向的方式，而这样也会造成 2 倍的参数冗余。如果非 MoE 模型，完全可以节省下这部分的显存，这显然不是追求极致性能 DeepSeek 所愿意看到的。再加上这种双向的 DualPipe 设计和实现都极为复杂，因此，这里可以**再次印证推理一**，DeepSeek要**将 All In MoE 的路线贯彻到底**。

### 2.6 总结和推论

上文介绍了很多DeepSeek的创新点，但在这些所有的创新点背后，有一条主线若隐若现，那就是模型系统的协同优化，我们得到了本文的第二条推理。

**推论二：模型系统协同优化，带来效果和性能的大幅提升。**

在 DeepSeek 的各种优化中，都可以看到模型系统协同优化的影子，具体如下：

- 在 MoE 模型结构的设计中，需要考虑专家的并行程度，来设计专家的数量。
- 为了实现专家的负载均衡，需要考虑设备和通讯的均衡。
- MLA是一种模型优化，但同样需要优秀的工程实现来支持。
- FP8 混合训练，为了实现极致的性能，也尝试了在模型的不同模块，采用不同的精度训练。
- DualPipe 是一种专门针对 MoE 结构设计的流水线并行。

而DeepSeek在招人的过程中，也会着重强调模型和工程的协同优化。职位描述中，明确写明：“要既懂算法，又懂系统；既能调精度，也能调性能；既能训练也要考虑推理部署”。这也说明DeepSeek从最开始的时候，就意识到**模型系统协同优化**的重要性，并通过人才招聘，**把这种特性写入了公司的基因里**。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/deepseek_hire.png" alt="deepseek_hire" style="zoom:40%;" />

到这里，还有一个疑问，别的厂商，也有过选择 MoE 结构的，为什么只有 DeepSeek 做出了最大的成果？这里就有了我们的推论三。

**推论三：坚定 All In MoE，带来了由 MoE 引发的一系列创新。**

有传闻，在 Llama 3 系列中，Meta 的团队同时训练了 dense 模型和 MoE 模型，但最终 MoE 模型因为训练不稳定，最终放弃了，只发布了 dense 模型。类似的，Mixtral 早期也是选择 MoE 结构，但后续又换成了 dense 模型。我一直有个观点，就是**“面对同样的问题，不同认知的人，会得出完全相反的结论。”** 当大家都知道 MoE 结构训练不稳定的时候，看不到 MoE 可以效果成本两开花，或者对于 MoE 能够稳定训练没有信心的人，会选择放弃这条路线，选择更稳定的 dense 模型。而坚信这一点的人，会选择坚定 All in MoE，然后死磕训练稳定性。很显然，DeepSeek 属于后者，这不光可以看出他们技术眼光的高超，也可以看出他们对于自己技术能力的自信（参考推理二）。不得不让人再次佩服。



除了本章上面提到的这些优化，DeepSeek 也还有一些其他的优化，比如 MTP（ Multi-Token Prediction）、继续扩大模型规模、采用更多的训练数据（V2 8.1T tokens，V3 14.8T tokens）等。最终，到DeepSeek V3 发布的时候，真的做到了第二象限的“最优选项”，一统江湖之势呼之欲出，只等 R1 最后点燃引爆全球的火种。

<img src="/Users/joywin/joywin/Technology-notes/papers/pics/pricevsperformance.png" alt="pricevsperformance" style="zoom:80%;" />



## 3. DeepSeek R1: 推理涌现，炸裂出圈

OpenAI推出的 O1 模型，通过长思考的方式，提升了模型在复杂推理任务的准确率之后，很多团队也对此进行了研究。一般而言，通过增加思维链（CoT）的推理路径，可以大大提升模型的效果。然而，大量标注 CoT 路径的数据成本非常高，而且，针对同一个问题，且答案唯一的情况下，CoT的推理路径往往可以不止一条。因此，通过大量标注 CoT 样本 + SFT的方式实现长思考模型，会面临成本高和泛化性差的问题。

### 3.1 推理涌现

面对推理任务需要的长思考路径，DeepSeek 的方式是，直接使用基于 GRPO 的强化学习方法进行优化。GRPO 是 DeepSeek 自己提出的一种强化学习方法，是对PPO方法的一种简化。关于GRPO的介绍，可以参考这篇文章：[无需RL基础理解 PPO 和 GRPO](https://mp.weixin.qq.com/s/YHoDl99fyNe7MP03BoRc6g)。

在 R1-Zero 和 R1 中，强化学习没有采用过程奖励模型（PRM），也没有采用一个单独的模型作为 reward model，DeepSeek的方式是，直接采用基于规则的奖励系统，具体而言主要有两种奖励：

1）准确性奖励：要求根据模型的回答准确通过规则。也就是说，规则只会检查模型最终输出的结果正确与否，而中间的推导过程，没有监督信号，也不会产生奖励。

2）格式奖励：要求模型将思考过程放在<think>和</think>标签之间。

通过如此简单的奖励信号，模型就通过自动探索，找到了通往正确答案的路径，这样的实现方法真的是既简洁，又十分有效，让人叹为观止。

同时，通过进一步的消融实验，可以发现，通过将已经训练好的 R1 模型蒸馏到更小的模型，效果会比直接在小的模型上使用强化学习效果更佳。

![distall_vs_rl](/Users/joywin/joywin/Technology-notes/papers/pics/distall_vs_rl.png)

因此，我们其实可以得到一个结论：DeepSeek V3模型能力已经足够强，**本身已经具备了解决复杂推理任务的能力**，但是会有多种解决方法，这些方法中有正确的，也有错误的。通过增加结果的准确性奖励，可以让模型从本已经具备的多种推理路径中，慢慢收敛到正确的推理路径上，从而出现了涌现智能。

而对于比较小，能力不够强的模型，本身还不具备解决这些复杂推理任务的能力，只给出结果的准确性奖励，模型就难以收敛到正确推理路径。当通过蒸馏的方式，把推理路径和正确结果一起注入模型的时候，模型的推理能力才得到了巨大的提升。因此，DeepSeek 也在原文中写道，“虽然蒸馏策略既经济又有效，但想要进一步推进智能的边界，仍然需要更强大的基础模型，和更大规模强化学习。” 

### 3.2 炸裂出圈

DeepSeek 的 V3 和 R1，不仅分别在对话模型和推理模型领域做到了最强，还全部开源了。从 Google 的搜索指数来看，DeepSeek（图中蓝线）的搜索量从24年底开始不断增长。从 V3 发布开始，对比 OpenAI （图中红线）的搜索指数，DeepSeek 先是增加到了 OpenAI 的一半左右。之后随着 R1 的发布，搜索量直接超越了 OpenAI 好多倍。可以看出业界对于 DeepSeek 的关注。

![v3_release](/Users/joywin/joywin/Technology-notes/papers/pics/v3_release.png)

![r1_release](/Users/joywin/joywin/Technology-notes/papers/pics/r1_release.png)





后期，DeepSeek 更是通过一周的开源，把一些 MLA、专家并行、DualPipe、FP8混合训练等重要技术全部开源，可以说开源的非常彻底。可以看的出，DeepSeek 是想把自己的方法同步到开源社区，然后彻底建立基于自己标准的开发生态。这一点，很容易让人想到早点的特斯拉，也是手握先进的技术专利，却选择免费开放给竞争对手。一方面，促进了整个行业的发展，另一方面，也体现了他们对于自己研发能力的绝对自信。



## 4. 汇总

现在，我们来尝试回答下开头的几个问题。

### DeepSeek不同版本是如何一次次进化，直至效果逆天的？

上文已经对相关的技术细节做了回顾，整体整理为一个表格如下所示。

|                                         | DeepSeek LLM    | DeepSeek V2            | DeepSeek V3                            |
| --------------------------------------- | --------------- | ---------------------- | -------------------------------------- |
| Total Params                            | 67B             | 236B                   | 671B                                   |
| Activated Params                        | 67B             | 21B                    | 37B                                    |
| Total Experts                           | -               | 2+160 experts          | 1+256 experts                          |
| Activated Experts                       | -               | 2+6 experts            | 1+8 experts                            |
| Tokens                                  | 2T              | 8.1T                   | 14.8T                                  |
| 主要贡献                                | 3条Scaling Laws | MoE架构、MLA、负载均衡 | DualPipe、FP8训练、无负载辅助损失、MTP |
| MMLU                                    | 71.3            | 78.4                   | 87.1                                   |
| 训练成本（H800 GPU  hours/每1T tokens） | 300K            | 172.8K                 | 180K                                   |

### 为什么众多大模型玩家中，脱颖而出的是他？对于我们又有哪些借鉴意义？

从上文的几个推论，我们其实可以看出，DeepSeek的大概的技术方向，在2024年1月就已经基本确定。比如，坚定长期主义、坚定 All In MoE、坚定模型系统协同优化策略、坚定开源。并且这些决策之间，还相互正反馈，直至后来爆发出惊人的能量。这么多决策中，如果其中的某一个选项没有选对，都不可能造就最终的 DeepSeek。可以说，DeepSeek在做大模型方面，每一个决策都选到了当前版本的正确答案。而且，这些决策都是在一年之前甚至更早，就都已经确定。不得不佩服DeepSeek的高认知和决策能力。

并且，为了能够彻底的贯彻自己的策略，DeepSeek 甚至选择不融资，避免收到来自投资人快速短期盈利的压力。可以说 DeepSeek 非常系统化的对自己的整个路径，做了非常深刻的规划。有人说，伟大是不能被计划的。当然，伟大不能被完全计划。但是，DeepSeek确实找到了那条通往伟大的最优路径。

DeepSeek对于我们的借鉴意义是什么？如果现阶段还要再有一个创业团队，去做大模型，真的可以直接抄DeepSeek作业，用长期主义的视角，坚定开源、坚定走模型系统协同优化的路线。作为AI行业从业者，也需要更多的从模型和系统两个角度，综合提升个人的能力。

### DeepSeek的成功，为未来大模型的进化提供了哪些方向？

个人斗胆做一些猜测如下：

- 短期内，在 Transformer 结构还继续是主要模型结构的情况下，70B 以上的模型，可以无脑选择 MoE结构。更何况 DeepEP 和 DualPipe 也已经开源了。
- Dense 模型在比较小的规模（maybe）还是有一定的用武之地。
- 在2.1节中，我们看到，FFN层的计算量占比为约62%，这部分已经通过MoE结构进行了优化，那接下来就是 attention 计算量的优化。这一点，DeepSeek 和 Kimi 也已经都发布了比较类似的工作。
- 再往长远的话，就要Transformer结构的创新了，更大的效果突破，需要更大的技术创新。



## 参考文献

- [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/pdf/2401.02954) 
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066) 
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/pdf/2405.04434)  
- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437) 
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)  
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/pdf/2112.06905) 
- [Mixtral of Experts](https://arxiv.org/pdf/2401.04088)  
- Deepseek v3 技术报告万字硬核解读 - 学车辆算法工程师的文章 - 知乎
  https://zhuanlan.zhihu.com/p/16323685381
- deepseek技术解读(1)-彻底理解MLA（Multi-Head Latent Attention） - 姜富春的文章 - 知乎
  https://zhuanlan.zhihu.com/p/16730036197
- deepseek技术解读(3)-MoE的演进之路 - 姜富春的文章 - 知乎
  https://zhuanlan.zhihu.com/p/18565423596
- 分析transformer模型的参数量、计算量、中间激活、KV cache - 回旋托马斯x的文章 - 知乎
  https://zhuanlan.zhihu.com/p/624740065
- Deepseek V3 预训练解读 - 大润发杀鱼工的文章 - 知乎
  https://zhuanlan.zhihu.com/p/15073492309
- https://martinfowler.com/articles/deepseek-papers.html
- Deepseek-v3技术报告-图的逐步解析-2-FP8混合精度 - 迷途小书僮的文章 - 知乎
  https://zhuanlan.zhihu.com/p/20807308858
- 简单聊聊Deepseek V3的FP8训练 - 机智流的文章 - 知乎
  https://zhuanlan.zhihu.com/p/15640684557
- 大模型涉及到的精度有多少种？FP32、TF32、FP16、BF16、FP8、FP4、NF4、INT8都有什么关联，一文讲清楚 - 一步留神的文章 - 知乎
  https://zhuanlan.zhihu.com/p/673708074
- DualPipe 深入浅出：没有分布式训练基础也能看懂的 DualPipe 全方位讲解 - 小天狼星不来客的文章 - 知乎
  https://zhuanlan.zhihu.com/p/27045651854

- https://huggingface.co/blog/ufotalent/cut-in-half-cn
- 【Deepseek 系列】V1/V2 /V3 /R1 技术演进之路速读 - 乞力马扎罗不说话的文章 - 知乎
  https://zhuanlan.zhihu.com/p/21752444414



