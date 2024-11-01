# Embedding



## E5

- 对于 zero-shot 设置，E5是第一个在 BEIR 检索基准测试上超越了强BM25基线的模型，且没有采用任何标记数据。
- 当微调时，E5在MTEB基准测试上获得了最佳结果。

- 精心策划的大规模文本对数据集（称为CCPairs）
  - 数据集包括Reddit 的(post, comment)对，Stackexchange 的(question, upvoted answer)对，英语维基百科的(entity name+section title, passage)对，科学论文的(title, abstract)和引用对，以及Common Crawl 网页和各种新闻来源的(title, passage)对。
  - 对Reddit和Common Crawl的数据应用简单的启发式规则进行过滤。例如，我们移除Reddit评论中过长（> 4096个字符）或得分低于1的评论，并移除具有高困惑度的网页文章。经过初步过滤后，我们最终得到约1.30亿对文本对，其中大多数来自Reddit和Common Crawl。
  - 一致性的数据过滤技术：首先在13亿对噪声文本对上训练模型，然后使用该模型对100万个随机段落进行排名。模型的预测应该与训练标签一致。这里我们根据手动检查的数据质量设置了k=2。经过这一步骤，我们得到了约2.7亿对文本对用于对比预训练。
  - 当在噪声数据集上训练时，神经网络倾向于先记住干净标签，然后逐渐过拟合噪声标签。类似的技术已被广泛用于去除数据集的噪声。
  - 原文中有图
- 数据的质量和多样性对于训练通用文本嵌入至关重要。
- SimCLR[10]推广的对比损失被发现比基于分类的损失[49, 14]更有效。

- 所有输入文本使用共享编码器，并通过添加两个前缀标识符“query:”和“passage:”来打破对称性，分别应用于q和d。这种不对称设计对于某些检索任务很重要
- 如何选择负样本。在这里，我们选择使用批次内的负样本[10]，其中一批其他对的段落作为负样本。我们发现，这种简单的策略使得训练更加稳定，并且在批量大小足够大时，其性能优于MoCo[25]等方法。

- 微调
  - 选择使用3个数据集的组合进行进一步训练：NLI 66（自然语言推理）、MS-MARCO段落排名数据集[8]和NQ（自然问题）数据集[30, 32]。
  - 损失函数

$$
\min D_{\mathrm{KL}}\left(p_{\mathrm{ce}}, p_{\mathrm{stu}}\right)+\alpha L_{\mathrm{cont}}
$$

- 消融实验
  - Batch size：将批次大小从 1K 增加到 32K 可使所有 6 个数据集都获得持续收益
  - 数据过滤：过滤掉低质量文本对，使用更少的优质数据可以比全量包含噪音的数据效果更好，且数据量级降低为1/4。



## GTE

- 效果

  - 只预训练的模型，在零样本文本检索任务中超越了BM25和E5模型，并且在MTEB基准测试中超越了许多监督模型。
  - 微调之后，110M的BERT模型就已经超过了openAI 的商业API
  - 在代码上有很好的效果

- 数据

  - 数据可以通过无需手动注释的方式广泛收集，从而有效地帮助训练文本表示模型
  - 包括网页（例如CommonCrawl，ClueWeb）、科学论文（例如arXiv，SemanticScholar）、社区问答论坛（例如StackExchange）、社交媒体（例如Reddit）、知识库（例如Wikipedia，DBPedia）以及代码仓库（例如StackOverflow，GitHub）。此外，我们还利用某些数据集中超链接的存在来促进文本对提取。

- Pooling方法：mean pooling of tokens representation

- 预训练

  - ∼800M text 的文本对，仅使用开源数据，没有采用任何过滤或清理方法。
  - 只使用in-batch负样本，大batch size很重要
  - 序列长度128，batch size 16384，steps 5w，5% warm up，
  - 超参数

  | Model    | Params | LR     | GPUs | BS    | Base LM                           |
  | -------- | ------ | ------ | ---- | ----- | --------------------------------- |
  | GTEsmall | 30M    | 3×10-4 | 2    | 16384 | microsoft/MiniLM-L12-H384-uncased |
  | GTEbase  | 110M   | 2×10-4 | 4    | 16384 | bert-base-uncased                 |
  | GTElarge | 330M   | 5×10-5 | 8    | 16384 | bert-large-uncased                |

- 微调阶段

  - ∼3M pairs 数据，大 batch size不再重要
  - 由于硬负样本已经能够提供对学习目标的可靠梯度估计，因此不需要大批量大小。
  - batch size 128，group size 16， seq len 512，lr 变小1个数量级

- 数据采样

  - 采用多项式分布来进行数据采样，从而缓解标签不平衡
    $$
    p_i=\frac{n_i^\alpha}{\sum_{j=1}^m n_j^\alpha}
    $$

  - 每一个batch内的数据，都是来自同一个数据源

- 损失函数改进
  $$
  L_{\mathrm{icl}}=-\frac{1}{n} \sum_{i=1}^n \log \frac{e^{s\left(q_i, d_i\right) / \tau}}{Z} \\
  
  \begin{aligned}
  where \ \ \ Z & =\sum_j e^{s\left(q_i, d_j\right) / \tau}+\sum_{j \neq i} e^{s\left(q_i, q_j\right) / \tau} \\
  & +\sum_j e^{s\left(q_j, d_i\right) / \tau}+\sum_{j \neq i} e^{s\left(d_j, d_i\right) / \tau}
  \end{aligned}
  $$

- 实验结果

  - Zero-shot Text Classification
    - 将输入文本直接转换为嵌入，将标签语言化为相应的文本，得到标签嵌入。输入嵌入和标签嵌入之间的距离通过它们的内积来测量，并且与输入文本具有最接近嵌入距离的标签被视为分类结果。
  - Unsupervised Text Retrieval
    - BEIR 是一个异构信息检索基准，包含不同格式、不同领域的检索任务。
  - Massive Text Embedding Benchmark

  - Code
    - 通过扩展数据量和计算资源，语言模型可以直接从代码标记序列中获取高质量的代码表示，而无需结合人类对代码结构信息的知识

- 消融实验

  - 数据越多，模型越大，效果越好。
  - batch size在1w达到饱和。
  - 20k步的时候基本收敛。
  - 2阶段训练效果优于单阶段的效果
  - 采样方式，多项式更好。a=0.5。
  - 改进的loss可以提升预训练和微调两阶段的效果。



## BGE

- C-Pack的资源包
  - C-MTP（中国大规模文本对）。一个包含1000万文本对的庞大训练数据集。我们的大多数数据集都是从庞大的网络语料库中精心挑选的，例如百度（类似维基百科的中文网络）、知乎（一个主要的中国社交媒体）、中国的主要新闻网站等。
  - C-MTEB（中国大规模文本嵌入基准）。是一个涵盖6个任务和35个数据集的中文文本嵌入综合基准测试。共有6组评估任务：检索、重新排名、STS（语义文本相似性）、分类、成对分类和聚类，涵盖了中文文本嵌入的主要方面。
  - BGE（BAAI通用嵌入）比先前的中文文本嵌入在C-MTEB上的性能提高了超过10%。
  - 训练方法：1）使用普通文本进行预训练，2）与未标记的C-MTP进行对比学习，以及3）与标记的C-MTP进行多任务学习。

- C-MTP：有个图
  - 未标记数据：在我们的工作中，我们采用了一种复合数据清洗策略来精炼原始数据。首先，整个数据经过一般过滤，移除非文本内容、重复和恶意内容。其次，通过语义过滤进一步处理数据，以确保文本对在语义上相关。在我们的工作中，我们使用了一个第三方模型：Text2Vec-Chinese8来评分每对文本的关系强度。我们经验性地选择了一个阈值为0.43，并丢弃得分低于该阈值的样本。通过这样的操作，从未标记的语料库中筛选出了1亿个文本对。
  - 标记数据：这些数据集涵盖了文本嵌入的不同能力，如检索、排名、相似性比较等。特别是，包括以下标记的数据集：T2*T*2 - 排名[60]、DuReader[20, 42]、mMARCO[8]、CMedQA-v2[65]、multi-cpr[31]、NLI-Zh 99、cmnli[62]和ocnlli[62]。总共有838,465对文本，包含了多样的问题和回答模式。尽管它比C-MTP（未标记）小得多，但大部分数据都是从人工注释中精心挑选的，因此确保了相关性的高可信度。
- 训练方法
  - 预训练：RetroMAE中提出的MAE风格方法，该方法简单但高效。
  - 



## Stella

https://zhuanlan.zhihu.com/p/655322183

- 避免模型的灾难性遗忘
  - Replay，即在训练数据中混合模型原有的训练数据和已有文本匹配数据
  - EWC,Elastic Weights Consolidation，openai的一篇论文，主要思想是对参数做一个约束，让新模型参数不至于偏离原模型太远。实际操作时，移除了需要在原始训练集计算的参数权重，转而使用固定值，权重设为了10，对于512-1024的position embedding，权重为0，因为这块参数是需要进行更新学习的，因此不做约束。

- **Train data：**

  1）开源数据(wudao_base_200GB、m3e和simclue)，着重挑选了长度大于512文本，小于512的文本以一定概率舍弃

  2）在通用语料库上使用LLM构造一批(question, paragraph)和(sentence, paragraph)数据

- **Loss function：**

  1. 对比学习损失函数，最经典的batch内负例，缩放系数为30
  2. 带有难负例的对比学习损失函数(分别基于bm25和vector构造了难负例)
  3. EWC(Elastic Weights Consolidation)
  4. cosent loss，用于训练带标签的文本对
  5. 每一种类型的数据一个迭代器，分别计算loss进行更新

- 多种数据集，对应多种loss，在实际操作时，我们采用了交替训练的方式，即每一步选取一种数据集计算对应的loss然后更新权重，下一步继续随机选取其他的数据集计算loss并更新权重。
  - 这样做有2个好处：1）缓解了类不平衡问题，每次都从不同数据中采样一个batch训练。2）避免了loss尺度不一致的问题，因为是单独计算互不影响。至于相加会不会更好，本人没做实验也不敢乱说。

- 通用编码模型训练技巧分享

  - dropout-1d

  dropout已经是深度学习的标配，我们可以稍微改造下使其更适合句向量的训练。 我们在训练时会尝试让每一个token-embedding都可以表征整个句子，而在推理时使用mean_pooling从而达到类似模型融合的效果。 具体操作是在mean_pooling时加入dropout_1d，torch代码如下：

  ```python
  vector_dropout = nn.Dropout1d(0.3)  # 算力有限，试了0.3和0.5 两个参数，其中0.3更优
  last_hidden_state = bert_model(...)[0]
  last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
  last_hidden = vector_dropout(last_hidden)
  vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
  ```



- 带有难负例的检索训练数据。约20万。https://huggingface.co/datasets/infgrad/retrieval_data_llm

- 对话重写数据集，总量约160万。https://huggingface.co/datasets/infgrad/dialogue_rewrite_llm



## Conan Embedding

- 预训练阶段
  - 了约7.5 亿个文本对，包含Wudao、Zhihu-KOL、SimCLUE等，数据分为标题-内容对、输入-输出对和问答对等。我们还发现，高质量的 LLM 指令调优数据（例如：提示-响应对）经过规则过滤和筛选后，可以提升 Embedding 模型的性能。此外，我们利用现有的文本语料库，使用 LLM 生成了一批数据。
  - 首先，通过文档提取和语言识别进行格式化处理；接着，在基于规则的阶段，文本会经过规范化和启发式过滤；然后，通过MinHash方法进行去重；在安全过滤阶段，执行域名阻止、毒性分类和色情内容分类；最后，在质量过滤阶段，文本会经过广告分类和流畅度分类，以确保输出文本的高质量。通过过滤，我们筛选了约 4.5 亿对数据，留存率约60%。
  - 在数据经过标准过滤后，我们使用bge-large-zh-v1.5模型对每一条数据进行评分，丢弃所有得分低于0.4的数据。通过评分，我们筛选了约 4 亿对数据，留存率约 89%。

- 微调阶段
  - 将任务分为两类：检索（Retrieval）和语义文本相似性（STS）。检索任务包括查询、正样本和负样本，经典的损失函数是InfoNCE Loss。STS任务涉及区分两段文本之间的相似性，经典的损失函数是交叉熵损失。在STS任务上，根据以往工作的结论，CoSENT损失略优于交叉熵损失。因此，我们也采用CoSENT损失来优化STS任务：

- 动态难负例挖掘训练
  - 难负例挖掘用于为query选择负样本。其思想是使用一个 teacher模型来找到与query有一定相关性但不如正样本相关的段落，从而使对比损失更难区分正例和负例。这些难负例应该比随机负例更难与正例区分，从而带来更高效和更有效的微调。
  - 随着训练的进行，每当模型权重更新时，当前权重下的模型对应的难负例就会变化。在数据预处理阶段挖掘的难负例在经过训练迭代后，就会变得不那么难了。
  - 对于每个数据，我们记录当前难负例与Query的平均分数。每 100 次迭代后，如果分数的 1.15 倍小于初始分数且分数绝对值小于 0.8时，我们就认为该负例不再困难，并进行新一轮的难负例挖掘。
  - 图3 展示了动态难负例挖掘与标准难负例挖掘的样本正负例Score - Steps 变化曲线。可以看到，随着步数的增加，Standard-HNM的负例评分不再下降，而是出现震荡，这表明模型对该批负例的学习已完成。而Dynamic-HDM在检测到负例学习完毕后，会进行难负例的替换。

- 跨GPU的Batch均衡训练

  - 为了更好地利用难样本，我们采用了跨 GPU 批次平衡损失 (CBB)。之前的方案通常在训练流程中随机的在每个Batch中分配一个任务。例如：在iter0中采样STS的样本，并使用STS对应Loss进行反向传播获取梯度并更新权重，而iter1中分配了Retri任务或者CLS任务，我们称之为顺序随机任务训练。这样训练几乎一定会导致单次的优化搜索空间与Embedding模型的全局优化搜索空间不一致，从而导致训练过程的震荡以及无法求得全局最优解。我们在之后的分析中展现了这一现象。

    为此，我们考虑在每次的Forward-Loss-Backward-Update的更新过程中都均衡的引入每一个任务，以此来获得稳定的搜索空间，并尽可能的缩小单次模型更新方向和全局最优解的一致性。因此，CBB策略不仅考虑了不同 GPU 之间的通信，还考虑了不同任务之间的通信，从而实现了更好的Batch均衡。如图4所示，为了在检索任务中利用更多难样本，我们确保 GPU（gpu0、gpu1、gpu2、gpu3）各自具有不同的负样本，同时共享相同的查询和相同的正样本。对于Retri任务，每个 GPU 计算对应Batch的Loss，然后将结果汇总到 gpu1 上。对于STS任务，在gpu4上，运行STS任务并获得对应Loss。最终汇总并计算当前Iter的合并 CBB Loss。对应公式如下：

- 实验细节

  - 与大多数Embedding模型一样，Conan-Embedding也采用BERT模型作为基础模型，并使用FC Layer将输出维度从1024扩展到1792。模型的参数量为326M。Conan-Embedding的最大输入长度为 512 个 token。此外，受到 OpenAI的text-embedding-v3的启发，我们还利用了多尺度表征学习（Matryoshka Representation Learning, MRL）技术来实现灵活的输出维度长度，提升模型表征性能和鲁棒性。对于 MRL 训练，表示维度配置为256、 512、 768、1024、1536 和 1792。

    为了提高效率，我们使用了混合精度训练和 DeepSpeed ZERO-Stage 1。

    弱监督预训练阶段，我们使用 AdamW优化器，初始学习率为 1e-5，Warmup设置为 0.05，Decay设置为 0.001。整个预训练过程使用了 64张华为Ascend 910B GPU，单次精调训练约消耗138 个小时。

    有监督精调阶段，检索任务的BatchSize设置为 4，STS 任务的BatchSize设置为32。我们使用了与预训练阶段相同的优化器参数和学习率。整个微调过程使用了 16 张华为Ascend 910B GPU，单次精调训练约消耗13 个小时。





## E5 Mistral

- 不再需要多阶段训练
- 仅使用合成数据和不到1000个训练步骤即可获得高质量的文本嵌入。
- 我们使用专有的LLM生成93种语言中各种文本嵌入任务的合成数据，涵盖数万个嵌入任务。具体来说，我们采用两步提示策略，首先提示LLM头脑风暴候选任务池，然后提示LLM根据给定任务从池中生成数据。为了涵盖各种应用场景，我们为每种任务类型设计了多个提示模板，并结合不同模板的生成数据以增强多样性。
- 与这些方法不同，我们的方法不依赖于任何未标记的文档或查询，因此可以生成更多样化的合成数据。
- 使用LLM可以生成hard -negnative

- 合成数据
  - 非对称任务：这个类别包括查询和文档在语义上相关但不是彼此的释义的任务。根据查询和文档的长度，我们将非对称任务进一步细分为四个子组：短长匹配、长短匹配、短短匹配和长短匹配。我们设计了一个两步提示模板，首先提示LLMs头脑风暴一系列任务，然后根据任务定义生成一个具体的示例。
  - 对称任务：对称任务涉及具有相似语义意义但表面形式不同的查询和文档。
  - 生成多语言数据，从XLM-R（Conneau等人，2020）的语言列表中抽样“{language}”的值，给予高资源语言更多的权重。任何不符合预定义JSON格式的数据在解析过程中都会被丢弃。我们还根据精确字符串匹配移除重复项。
- 训练：
  - query给出task_definition的prompt，给每个任务细粒度的任务定义。doc保持不变。
  - 温度系数：0.02。EOS方式获取embedding
  - 训练：Mistral-7b，1 epoch，LoRA rank 16。batch 2048，lr 1e-4，seq len 512，32 V100 for 18h。
  - 合成数据：50w合成数据，使用了15w的instruction。涉及93种语言。
  - 总数据：合成数据+13个公开数据集抽样，共1.8M。对于没有hard ngeatives的样本，使用mE5 base挖掘top 100的hard negatives。
  - 生成语言建模和文本嵌入是同一枚硬币的两面，两个任务都需要模型对自然语言有深入的理解。给定一个嵌入任务定义，一个真正的鲁棒LLM应该能够自己生成训练数据，然后通过轻量级的微调转换为嵌入模型。
  - 对于LLM，对比学习pretrain阶段不是必需的
  - 



## VLM2VEC







## M3CSR

- 步骤
  - 预处理，将content embedding聚类为兴趣中心，并建立一个兴趣中心的embedding table，使得每个兴趣中心都是可训练的
  - 一个modal encoder，可以获取每个模态的embedding
  - 







- 一些启发：
  - 现在的文本embedding基本都采用预训练+微调的2阶段训练方式
  - 预训练阶段，数据的分层比较重要，即同一batch中的样本，需要来自同一数据来源（我们的场景下，可以是同一分类）
  - GTE中对infoNCE loss有一个改进，可以在使用同样batch的情况下，有更多的负样本，可以借鉴
  - batch size在预训练阶段比较重要，在微调阶段不再重要。微调阶段负样本选择比较重要，方法主要有基于retrieval model和BM25的方法
  - 对于数据不平衡情况，多项式采样的方法比随机采样和分层采样效果更好
  - 评估方面，除去linear probe之外，还可以采用zero shot方法，即将text 和label都通过embedding模型，然后通过计算text 和label 相似度的方法进行分类。（不再需要训练线性层）
  - 避免灾难性遗忘，除了使用repaly（混合模型原有的训练数据和已有文本匹配数据）之外，还可以使用EWC方法

- Improving Text Embeddings with Large Language Models
  - 采用GPT4生成了合成数据，和有标签数据，一共180w数据
  - 使用 [EOS] vector 作为 embedding，InfoNCE 作为 loss 进行训练
  - 模型采用 Mistral-7B，batch size 2048，32V100训练18小时
  - MTEB +2%
  - 缺点：embedding维度比较大，4096

- NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models
  - 和上面文章同样采用了InfoNCE 作为 loss 进行训练
  - 使用了latent attention layer 来获取 embedding
  - 不同的embedding获取方式：latent attention layer > mean pooling > EOS
  - 训练方式：一阶段，在检索数据集上进行对比学习，二阶段，将非检索数据混合到指令微调中。
  - 当前sota

- JINA CLIP: Your CLIP Model Is Also Your Text Retriever
  - CLIP-style 模型在text-text 任务中效果比纯文本任务上训练的模型效果差
  - 设计了一个3阶段-多任务的训练方式
  - 多任务：text-text 和 text-image 一起进行训练
  - 多阶段：先训练短文本，再训练长文本
  - 可以在 text-text 和 text-image 上均得到较好效果

- NoteLLM: A Retrievable Large Language Model for Note Recommendation
  - 和我们的当前任务最相近
  - 2 个任务：生成式对比学习（Generative-Contrastive Learning）、协同监督微调（Collaborative Supervised Fine-Tuning）
  - 生成式对比学习：LLaMA 2作为backbone，使用特殊token [EMB] 获取embedding
  - 协同监督微调：标题和标签生成任务（text generation）。有点类似我们的prop
  - 评估指标：recall@k 
  - 效果：一周的ab实验，跟之前的SentenceBERT基线相比，NoteLLM的点击率提高了16.20%，评论数量增加了1.10%，平均每周发布者数量（WAP）增加了0.41%。结果表明将LLM引入i2i推荐任务可以提高推荐性能和用户体验。此外，还观察到单日对新笔记的评论数量显着增加了3.58%。这表明LLM的引入有利于冷启动。NoteLLM最终推全上线。



## Arctic-Embed

- Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models
- 模型参数：22 to 334 million
- 同等参数情况下，在 MTEB Retrieval Average nDCG@10上，效果超过BGE、GTE、E5
- arctic-embed-l 超越了闭源模型  Cohere’s embed-v3 和 Open AI’s text-embed-3-large
- 相比于扩大数据量和batch size，数据采样（质量）更关键。
- 基于难负样本的query生成效果增益很大。（而不是同时生成query和负样本）

#### 三、Arctic Embed

- 训练方法：
  - 第一阶段，大规模预训练，采用in-batch负样本。从20亿数据中，过滤得到了3.08亿数据。
  - 第二阶段，微调，采用难负样本。100w pair对。
- 采用 CLS token 而不是 mean pooling 获取embedding。在 STS 上，~2.5 NDCG@10 
- 数据清洗：对文本质量和文本配对关系均进行了清洗
  - 文本质量：语言过滤、文档长度过滤，单词长度过滤、特殊符号过滤、省略号过滤、非字母过滤、困惑度过滤、N-gram重复过滤、停用词过滤、要点行过滤、黑名单过滤、短行过滤、数字行过滤、大写行过滤。
  - 文本配对关系：使用 fastText 计算相似度，然后取0.3阈值

- 数据质量比数量更重要：去除了 NLI, MEDI, WikiAnswers, and SQuAD 这些作用很小的数据。

- 直接将所有数据样本进行混合，不是一个很好的策略。每个数据集都单独进行了实验，然后根据这些实验，对数据集进行选择和组合。

- 生成query的时候，将负样本加入到LLM的输入中非常重要。

- 可调节的难负样本挖掘：

  - 使用一个已有的embedding 模型来对每个query的负样本进行打分。使用一个阈值而不是topk的策略保留负样本。选择相似度在阈值之间的样本。

  - $$
    s^{\prime}=\left[s_i: R_{\min } \leq s_i \leq R_{\max }\right]
    $$

  - 实际操作中，挖掘了 top 100，然后使用一个上限阈值。

- 使用课程学习（curriculum learning），通过调整负样本难度的顺序，也可以进一步提升效果。

####  四、训练方法

- 在信息检索任务上预训练过的模型，比通用的预训练模型效果更好。
- 预训练阶段，使用大规模数据和大的batch size，学习率和学习率的 schedule 对结果影响很大。
- 更长的文本对于学习有增益
- 数据源分层：预训练期间，单个批次中，使用的都是同一来源的数据
- 微调阶段，没有使用warmup策略，和预训练阶段相同的 lr decay策略，文本长度进一步变长，来到了512。 
- 微调阶段，停止使用in-batch 负样本。1个正样本和10个负样本。

#### 五、提升训练效率

#### 六、实验结果

#### 七、消融实验

- 预训练，数据分层很重要。其他涨点的因素：大的batch和seq len。
- 微调阶段，负样本难度的选择，阈值太高或者太低都会影响性能。最终合适的是0.5，而不是0.8或者0.4。



| Training State                 | Batches | Doc/ Batch | Doc Max Length | Elapsed | Doc/ Sec |
| ------------------------------ | ------- | ---------- | -------------- | ------- | -------- |
| Large Scale In-Batch Negatives | 18,798  | 16,384     | 256            | 17h3m   | 5,018    |
| Smaller Scale Hard Negatives   | 7,845   | 5,632      | 512            | 7h40m   | 1,601    |



| Variant | Pre Batch | Pre LR | Finetune Batch | Fine LR |
| ------- | --------- | ------ | -------------- | ------- |
| xs      | 24,576    | 6e-4   | 768            | 4e-5    |
| s       | 32,768    | 5e-4   | 1,024          | 4e-5    |
| m       | 16,384    | 2e-4   | 512            | 1e-5    |
| m-long  | 12,288    | 1e-4   | 512            | 1e-5    |
| l       | 12,480    | 1e-4   | 512            | 9e-6    |

- 训练向量(embedding)模型的6个经验总结 - 车中草同学的文章 - 知乎
  https://zhuanlan.zhihu.com/p/697372928





- - 



## Stella

- 长文本（支持到1024）
  - 使用苏神的层次分解位置编码进行初始化512-1024的position embedding。
  - 构造长文本。
    - 对于开源数据中的长数据，将其切分成句子，然后依次和query计算集合相似度，相似度最高的句子如果处于passage后半段那么就纳入训练数据，否则就以一定的比例舍弃。
    - 找一些长度大于512的无监督文本块，让LLM来生成相关query。抽取文本块后面的几个句子，然后模型生成对应的query。
  - 避免灾难性遗忘
    - Replay，即在训练数据中混合模型原有的训练数据和已有文本匹配数据。
    - EWC，Elastic Weights Consolidation，openai的一篇论文，主要思想是对参数做一个约束，让新模型参数不至于偏离原模型太远。

- Loss function：

  1. 对比学习损失函数，最经典的batch内负例，缩放系数为30
  2. 带有难负例的对比学习损失函数(分别基于bm25和vector构造了难负例)
  3. EWC(Elastic Weights Consolidation)
  4. cosent loss，用于训练带标签的文本对

  我们可以发现这次训练有多种数据集，对应多种loss，在实际操作时，我们采用了交替训练的方式，即每一步选取一种数据集计算对应的loss然后更新权重，下一步继续随机选取其他的数据集计算loss并更新权重。

## E5

- 摘要
  - 使用弱监督信号，对比学习的方法进行训练。
  - 可以用于广泛的各种文本表征任务。
  - 在多个数据集上表现很好。

- CCPairs
  - 从网络文本中获取文本对，经过简单的初始规则过滤，得到了1.3B的文本对
  - 对数据进行了激进的过滤，先用高噪声数据训练模型，然后对标签进行清洗。得到了～270M文本对。
  - 神经网络的记忆行为：对嘈杂的数据集进行训练时，神经网络倾向于首先记住干净的标签，然后逐渐过度拟合嘈杂的标签。

- 





|                    | E5                              | GTE                                      | BGE                                          | Arctic                            |
| ------------------ | ------------------------------- | ---------------------------------------- | -------------------------------------------- | --------------------------------- |
| 预训练数据         | CCPairs（～270M）               | 788M                                     | 100M                                         | 308M                              |
| 清洗策略           | dataset 自举，保留label在top2中 | 无清洗                                   | 文本质量过滤、文本配对过滤（相似度阈值0.43） | 文本质量过滤、文本配对过滤        |
| pooling            | mean pooling                    | mean pooling                             |                                              | CLS token                         |
| 预训练loss         | InfoNCE                         | 改进 InfoNCE                             |                                              | InfoNCE                           |
| 温度系数           | 0.01                            | 0.01                                     |                                              |                                   |
| 文本前缀           | 增加query、passage              |                                          | 增加前缀prompt                               |                                   |
| batch size         | 32, 768                         | 16384                                    | 19,200                                       | 12,480                            |
| seq len            | 128                             | 128                                      |                                              | 256                               |
| learaning rate     | 1e-4                            | 5e−5                                     |                                              | 1e-4                              |
| warm up            | 1000 steps                      | 5% (2500 steps)                          |                                              | 几百步                            |
| Initialization     | bert-large-uncased-wwm          | bert-large-uncased                       |                                              |                                   |
| Total steps        | 20k steps (2.5 epochs)          | 50k steps (1 epoch)                      |                                              | 1 epoch                           |
| BEIR (nDCG@10)     | 44.2                            | 44.6                                     |                                              |                                   |
| MTEB               | 56.6                            | 59.3                                     |                                              |                                   |
|                    |                                 |                                          |                                              |                                   |
| 微调数据           | NLI, MS-MARCO, NQ               | ∼3M                                      | 838k                                         | 1M                                |
| 微调loss           | 蒸馏KL散度 + InfoNCE            | 改进 InfoNCE                             |                                              | InfoNCE                           |
| hard negative      | NLI 中的矛盾句子                | 从 retrieval system topk去除正样本后采样 |                                              | 从retrieval system取上下阈值      |
| 其他方式           | 从一个CE进行知识蒸馏            | 数据多项式采样                           |                                              | 基于难负样本的query生成，课程学习 |
| batch size         | 256                             | 128                                      |                                              | 512                               |
| learaning rate     | 1e-5                            | 5e−6                                     |                                              | 9e−6                              |
| seq len            | 192                             | 512                                      |                                              | 512                               |
| warm up            | 400 steps                       |                                          |                                              |                                   |
| Total steps        | 3 epochs                        | 1 epoch                                  |                                              |                                   |
| hard negatives num | 7                               | 15                                       |                                              | 10                                |
|                    |                                 |                                          |                                              |                                   |
| BEIR (nDCG@10)     | 50.0                            | -                                        |                                              |                                   |
| MTEB               | 61.4                            | 63.1                                     |                                              |                                   |



## NoteLLM

- 生成式对比学习（Generative-Contrastive Learning）、协同监督微调（Collaborative Supervised Fine-Tuning）

- Loss
  $$
  L=\frac{L_{c l}+\alpha L_{g e n}}{1+\alpha} \\
  \alpha = 0.01
  $$

- 维度128，标题20tokens，内容128tokens。batch size 64，包含128笔记。对比学习温度系数3



​                ● Improving Text Embeddings with Large Language Models

​                ○ 采用GPT4生成了合成数据，和有标签数据，一共180w数据

​                ○ 使用 [EOS] vector 作为 embedding，InfoNCE 作为 loss 进行训练

​                ○ 模型采用 Mistral-7B，batch size 2048，32V100训练18小时

​                ○ MTEB +2%

​                ○ 缺点：embedding维度比较大，4096

​                ● NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models

​                ○ 和上面文章同样采用了InfoNCE 作为 loss 进行训练

​                ○ 使用了latent attention layer 来获取 embedding

​                ○ 不同的embedding获取方式：latent attention layer > mean pooling > EOS

​                ○ 训练方式：一阶段，在检索数据集上进行对比学习，二阶段，将非检索数据混合到指令微调中。

​                ○ 当前sota

​                ● JINA CLIP: Your CLIP Model Is Also Your Text Retriever

​                ○ CLIP-style 模型在text-text 任务中效果比纯文本任务上训练的模型效果差

​                ○ 设计了一个3阶段-多任务的训练方式

​                ○ 多任务：text-text 和 text-image 一起进行训练

​                ○ 多阶段：先训练短文本，再训练长文本

​                ○ 可以在 text-text 和 text-image 上均得到较好效果

​                ● NoteLLM: A Retrievable Large Language Model for Note Recommendation

​                ○ 和我们的当前任务最相近

​                ○ 2 个任务：生成式对比学习（Generative-Contrastive Learning）、协同监督微调（Collaborative Supervised Fine-Tuning）

​                ○ 生成式对比学习：LLaMA 2作为backbone，使用特殊token [EMB] 获取embedding

​                ○ 协同监督微调：标题和标签生成任务（text generation）。有点类似我们的prop

​                ○ 评估指标：recall@k 

​                ○ 效果：一周的ab实验，跟之前的SentenceBERT基线相比，NoteLLM的点击率提高了16.20%，评论数量增加了1.10%，平均每周发布者数量（WAP）增加了0.41%。结果表明将LLM引入i2i推荐任务可以提高推荐性能和用户体验。此外，还观察到单日对新笔记的评论数量显着增加了3.58%。这表明LLM的引入有利于冷启动。NoteLLM最终推全上线。



## NoteLLM-2

- 摘要：
  - 提出了一个端到端的训练方法，可以使用现有任意的LLM和visual encoder
  - 为了解决视觉信息容易被忽略的问题，采用了 multimodal In-Context Learning（mICL）和late fusion





## MISSRec

- 摘要

  - 基于ID的模型问题：稀疏性、冷启、不同域之间难以联合优化
  - MISSRec：多模态模型、包含预训练和微调阶段、鲁棒且通用
  - 结构：encoder、decoder、动态fusion模块
  - 损失函数：对比学习损失

- 引言

  - 基于ID模型的问题：
    - 数据稀疏性、对于热门item会有bias
    - 不同域之间难以联合优化

  - 论文关注的2个问题
    - 多模态信息如何影响序列推荐？
    - 如何在序列推荐中更好利用多模态信息？
  - 多模态信息应用于序列推荐的2个挑战
    - 每个 item 的多模态协同是用户依赖且动态变化的，用户可能出于不同的原因与同一 item 互动。而且不同的模态对用户兴趣的影响程度也不同。即使是对于相同的 user-item 组合，这种模式也可能随时间和情境而变化，这使得设计有效的多模态融合变得困难。
    - 信息冗余可能会掩盖用户的核心兴趣。交互序列通常表现出兴趣分布不平衡的特点。用户的序列中通常包含大量的同质化项目，例如日常必需品，而其他信息丰富的项目，如体育用品，则可能稀疏出现。如果我们在用户行为建模中平等对待所有交互，这将导致对某些类型项目的过度强调，而对其他项目的关注不足。
  - MISSRec如何处理上述的挑战
    - 为了弥合通用领域与推荐系统之间的语义鸿沟，我们引入了多模态特征适配器，从而提取通用多模态特征的个性化语义。
    - 为了探索多模态协同作用，设计了一个基于Transformer的编码器用于处理多模态序列。与item级别的静态融合（如向量加法）相比，能够自适应地捕捉每个项目中有用的模态线索，以实现个性化。在候选item方面，也引入了一种轻量级的动态融合策略来生成特定于用户的item表示。
    - 为减轻信息冗余对序列建模的负面影响，引入了一个兴趣发现模块来挖掘用户间的全局多模态兴趣。通过自适应聚类将item与相关兴趣关联起来，从而将多模态序列转换成一系列兴趣token。随后，我们提出了一个兴趣感知的Transformer解码器，它以编码器输出序列为key和value，去重后的用户兴趣作为query，来掌握item-模态-兴趣的关键模式，从而实现精确而全面的序列表示。
  - 主要贡献
    - 强调了利用多模态信息进行 SR 的重要性和挑战，并提出了一种有效的预训练及高效的迁移学习框架。
    - 为了捕捉上下文和动态的多模态协同作用，我们设计了一个基于Transformer的情境编码器用于多模态序列建模，并采用了一个轻量级的动态融合模块来生成适应用户的候选item表示。
    - 我们引入了一个多模态兴趣发现模块，在此基础上构建了一个兴趣感知解码器，以建模item-模态-兴趣关系，从而实现更好的序列表示。

- 相关工作

  - 序列推荐
    - 马尔可夫链
    - 深度学习方法：CNN、RNN、MLP based
    - 兴趣建模：attention or 聚类
    - 训练策略：时间感知任务、对比学习目标函数
    - 采用多模态内容：CSAN、MM-Rec、MMMLP
  - 



## HLLM

### 摘要

- 关于LLM的3个问题
  - LLMs的真正价值（通常认为预训练权重包含了世界知识）；
  - 微调对于推荐任务的必要性；
  - LLMs是否能够在推荐系统中像在其他领域一样展现出同样的可扩展性优势。
- HLLM（Hierarchical Large Language Model）的两级模型包括：
  1. **Item LLM（第一层）**：专门用于从item的详细文本描述中提取丰富的内容特征。它将项目的文本信息（如标题、标签等）作为输入，并输出一个压缩后的向量表示，即item的embedding。
  2. **User LLM（第二层）**：它使用第一层提取的项目特征来建模用户兴趣，并预测用户的未来行为。它接受用户历史交互的项目特征序列作为输入，并预测下一个用户可能感兴趣的item的embedding。
- 效果
  - 两个大规模数据集PixelRec和Amazon Reviews上的评估表明，HLLM实现了最先进的结果，远远超过了传统的基于ID的模型。
  - 在线A/B测试中，HLLM展示了显著的增益，验证了其在现实世界推荐场景中的实际影响。

### 导言

- 将用户行为历史，使用文本的形式输入模型的问题
  - 输入很长（和基于id的模型相比）
  - 相同的时间跨度，序列长度更长
  - self- attention的平方时间复杂度，使得计算量快速上升
  - 一个item需要多次推理，是的效率很低

- 本文的贡献 
  - 提出了新的HLLM结构，超过了经典的基于ID的方法，并且在训练和服务方面都很高效。
  - HLLM有效地将LLM预训练阶段编码的世界知识转移到推荐模型中，尽管如此，针对推荐目标的任务特定微调是必不可少的。
  - HLLM表现出卓越的可扩展性，随着数据量和模型参数的增加，性能持续提。
- 













- 推荐系统中如何做 User Embedding？ - 王鸿伟的回答 - 知乎
  https://www.zhihu.com/question/336110178/answer/823523924



















