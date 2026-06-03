# 代码视角详解沐神新作 Higgs Audio V2

大家好，我是 JOYWIN。前段时间沐神在B站和 github 发布了一个 audioLLM 模型，Higgs Audio V2。让 text LLM 在能读能写的基础上，增加了理解和生成 audio 的能力，变得能听也能说。一下子就被这个工作吸引了，感觉这是个非常 promising 的方向。让大模型在多模态理解和生成上，进行端到端训练，无论是技术研究还是应用，感觉都还有非常多的空间可以挖掘。（咱就是说，咱也是出息了，技术判断力和沐神一样了，真是会给自己脸上贴金，bushi）

然而，沐神虽然做了一个视频对模型进行了一些简单介绍，但是非常多的细节并没有详细讲解。同时，这个工作没有 paper，只有开源代码。为了能够详细理解模型的实现细节，我把模型代码库扒了一遍，通过单步调试，摸清了整个模型的细节。既然沐神没有时间讲解，希望通过本文可以让你对这个很棒的工作，有一个详细的理解。争取做成全网最全的 Higgs Audio V2 解读。

# 1. 为什么需要 audioLLM

在端到端的 audioLLM 之前，如果需要以 audio 作为媒介和 LLM 进行交互，往往需要一个 ASR+LLM+TTS 的多阶段模型。这样的方案，首先模型优化角度，没有办法做梯度的端到端回传，前面阶段模型的错误，没法通过后面阶段的 label 进行学习和纠正，使得模型的上限会较低。同时，在 audio 中，除了语言的语义信息之外，还有很多的信息，是通过语音、语调、音色、背景音等声学信息进行表达的。这些信息会在多阶段过程中被去除，使得信息有损。同时，多阶段的模型，也会存在延迟更长的问题，影响用户体验。因此，使用端到端的 audioLLM 可以有以下优势：

1. 端到端的模型训练，更利于梯度回传和模型优化。
2. 对音频进行更精细复杂的理解和生成，包括 auido 中包含的场景、人物、情绪等等。
3. 可以做到延时更低，提升用户体验。

通过一个端到端的模型，如果可以完成全部的“听说读写”，相当于用一个固定、简单的框架，处理所有的问题。然后使用海量的数据进行训练，通过 scaling law 大力出奇迹。这个非常符合 LLM 的训练风格，也符合 Rich Sutton 在 The Bitter Lesson 中的观点，也就是真正有效的路径都是那些简单通用但能承载更大算力投入的方案，而不是通过方法复杂化，追求人类先验知识的引入。

技术方向理清楚之后，就是如何实现一个端到端的 audioLLM 模型。具体包含 tokenizer、模型结构、数据构建，下文依次进行描述。

# 2. Tokenizer

原始的音频信号，在进入模型之前，需要通过 tokenizer，将原始信号转换为模型可以理解的 token。因此，作为原始信息和模型之间的桥梁，tokenizer 起到了一个关键的作用。一个好的 tokenizer，需要能够尽可能多的表征原始信号中的信息，同时将其离散化，从而更好的利于下游模型的学习。

## 2.1 预备知识

在阅读这个代码之前，笔者对于 audio 模型，完全没有了解，也不知道一段音频，在计算机中是如何表征和存储的。因此，作为一个语音新手，为了照顾同样背景的小伙伴，首先补充一点语音方面的基础知识。

### 2.1.1 语音的表征

首先，计算机中，如何表征一段音频。其实也比较简单，我们先采样一段1s中的音频，利用 Higgs Audio V2 中给出的代码，将其读取出来，代码如下：

```python
import librosa
audio_path_or_wv = 'higgs-audio/examples/voice_prompts/mabaoguo_1s.wav'
wv, sr = librosa.load(audio_path_or_wv, mono=True, sr=None)
print(wv.shape, sr)
# (44100,) 44100
print(wv)
# array([-0.08239746, -0.08969116, -0.09747314, ...,  0.00610352,
#         0.00842285,  0.01126099], shape=(44100,), dtype=float32)
```

其中，wv 就是读取得到的 audio，sr为采样率（sample rate）。可以看到，默认 sr 为 44100 时，1s的音频对应的就是一个长度为 44100 的向量。后续模型中，使用的默认的 sample rate 为 24000，因此，需要对音频进行重采样，使其 sr = 24k。

```python
if sr != self.sampling_rate: # self.sampling_rate == 24000
    wv = librosa.resample(wv, orig_sr=sr, target_sr=self.sampling_rate)
```

这样我们就得到了 tokenizer 的原始输入，一段以向量形式表征的一段音频。

### 2.1.2 语音的语义理解模型

在文本领域，我们可以通过一个语义理解模型（比如BERT），对文本的语义进行理解。同样的，在语音领域，也有类似的模型。在 Higgs Audio V2 中，使用的语义理解模型是 hubert，简单可以将其理解为一个语音版本的 BERT。模型的整体结构如下所示。

```python
(semantic_model): HubertModel(
    (feature_extractor): HubertFeatureEncoder()
    (feature_projection): HubertFeatureProjection()
    (encoder): HubertEncoder()
```

可以看到，HubertModel 一共包含3个部分，分别是 feature_extractor、feature_projection 和 encoder。其中 feature_extractor 对原始的语音信号进行一维卷积，从而使其长度变短，但是单个位置的维度变宽。feature_projection 就是一个 dense 结构，对表征进行一个线性映射。encoder 就是类似 BERT 的 12 层 transformer block。具体每个模块的结构如下：

feature_extractor 模型结构

```python
(feature_extractor): HubertFeatureEncoder(
      (conv_layers): ModuleList(
        (0): HubertGroupNormConvLayer(
          (conv): Conv1d(1, 512, kernel_size=(10,), stride=(5,), bias=False)
          (activation): GELUActivation()
          (layer_norm): GroupNorm(512, 512, eps=1e-05, affine=True)
        )
        (1-4): 4 x HubertNoLayerNormConvLayer(
          (conv): Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
          (activation): GELUActivation()
        )
        (5-6): 2 x HubertNoLayerNormConvLayer(
          (conv): Conv1d(512, 512, kernel_size=(2,), stride=(2,), bias=False)
          (activation): GELUActivation()
        )
      )
    )
```

从模型结构可以看出，feature_extractor 一共有7个一维卷积层，第一层的 stride 分别是5，其余 6 层的 stride 为2，则原始的音频长度会被降低为 $5*2^6=320$ 倍。在 HubertModel 中， 会单独采用一个新的 semantic_sample_rate，原始音频会被重采样为 16000。因此，经过 feature_extractor 之后，1s的音频的长度会变为 16000/320 = 50 。同时，由于第 0 层，卷积的维度为 512。因此，假设输入batch=1，维度为 (1, 1, 16000)，则输出的维度为(1, 50, 512)。

```python
(feature_projection): HubertFeatureProjection(
      (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (projection): Linear(in_features=512, out_features=768, bias=True)
      (dropout): Dropout(p=0.0, inplace=False)
    )
```

feature_projection 很简单，一个 Linear 层，维度变换，从 512 变为 768维（768？太熟悉了！就是BERT 的 hidden size！）。输出的维度变为 (1, 50, 768)。

```python
(encoder): HubertEncoder(
      (pos_conv_embed): HubertPositionalConvEmbedding(
        (conv): ParametrizedConv1d(
          768, 768, kernel_size=(128,), stride=(1,), padding=(64,), groups=16
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _WeightNorm())))
        (padding): HubertSamePadLayer()
        (activation): GELUActivation()
      )
      (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
      (layers): ModuleList(
        (0-11): 12 x HubertEncoderLayer(
          (attention): HubertSdpaAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (feed_forward): HubertFeedForward(
            (intermediate_dropout): Dropout(p=0.1, inplace=False)
            (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
            (output_dense): Linear(in_features=3072, out_features=768, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True))))
```

encoder 也很简单，就是一个12层的 BERT base，增加了一个 pos_conv_embed，输出维度和输入一样，还是(1, 50, 768)。

在经过 HubertModel 之后，1s的音频就会变为 50 个 audio token。token 数还是太多了，这里再次进行了一次2倍的下采样，最终得到的输出维度为 (1, 25, 768)。

```python
if self.semantic_downsample_factor > 1: # 其中，self.semantic_downsample_factor == 2
    target = target[:, :: self.semantic_downsample_factor, :]
```

### 2.1.3 RVQ

RVQ（Residual Vector Quantization）是对传统矢量量化（VQ）的改进。传统 VQ 只用一个码本对整个输入向量进行近似，而 RVQ 引入了残差学习机制：

1. 第一层：用第一个码本对原始输入向量 $x$ 进行粗略量化，得到近似值 $q_1$。
2. 计算残差：$r_1=x−q_1$ 
3. 第二层：用第二个码本对残差 $r_1$ 进行量化，得到 $q_2$，更新残差 $r_2=r_1−q_2$ 
4. 重复上述过程，共使用 $N$ 个码本（即 $N$ 层量化）
5. 最终重建：$\hat{x}=q_1+q_2+\cdots+q_N$

这样，每一层都在“修正”前一层的误差，逐步逼近原始信号。

代码如下：

```python
class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(**kwargs) for _ in range(num_quantizers)])

    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = 0.0
        residual = x
        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses
```

## 2.2 双流 tokenizer

有了上述的基础知识，就可以很容易的看懂下面的 tokenizer 的结构了。由于 audio 中会同时包含文字语义信息和声学信息，因此，Higgs Audio V2 采用了语义和声学的双流 tokenizer 对原始输入进行建模。tokenizer 的整体结构如下所示。

![dual_tokenizer](/Users/joywin/joywin/Technology-notes/week_report_2025/pics/dual_tokenizer.png)

其中 Semantic Teacher 就是 2.1.2 中的 HubertModel，并且模型参数是冻结的。同样取时长为 1s 的 audio，先走上面的一条语义支路，如 2.1.2 中所述，输入 $X$ 会从 （1, 1, 24000）变为维度为（1, 25, 768）的 $S$，其中 25 表示序列长度为25，768表示 hidden size。接下来的 Semantic Encoder 会对 $S$ 进行一个变换，但是维度保持一致，得到维度为（1, 768, 25）的语义表征 $h_S$ 。然后看下面的声学支路，Acoustic Encoder 是一个和 HubertModel 中的 feature_extractor 类似的结构，也是由多个一维卷积层组成，经过多次卷积之后，得到一个维度为（1, 256, 25）的声学表征 $h_A$ 。将语义表征 $h_S$ 和声学表征 $ h_A$ 拼接之后，得到了一个维度为（1, 1024, 25）的最终表征。这个表征，会经过1个 RVQ 模块。 Higgs Audio V2 中采用的 RVQ 的码本大小为 8 * 1024，因此经过量化之后，会得到一个（1, 8, 25）的量化表征。其中，8表示8层码本，25 表示 25 个 token。再接下来就是通过 Dense 线性变换之后，通过 Semantic Decoder 和 Acoustic Decoder 进行反卷积，最后得到重建之后的 $\hat{S}$ 和 $\hat{X}$ 。然后进行重建损失的计算。

可以看到，通过双流的设计，可以对音频中的语义和声学信息分别进行单独建模。一般而言，由于音频中的语义信息的重要性要大于声学信息，因此，表征中，语义信息占据的维度（768）也大于声学的维度（256）。通过 RVQ 的残差量化方式，可以使得在码本大小较小（8*1024）的情况下，能够对信息进行足够复杂的表示。此外，对比其他的音频生成tokenizer，这个 tokenizer 还有如下的优势：

- 低帧率：25 帧每秒的帧率相较于许多基线模型降低一半，同时保持高质量音频。
- 统一 24 kHz 训练：将语音、音乐和声音事件片段混合在一个模型中，捕获语义和声学细节，极大促进音频语言模型的训练。
- 快速推理：避免扩散步骤，编码器/解码器快速处理批次，适用于实时或大规模任务。

# 3. 模型结构

接下来我们看模型的结构。模型的主体结构如下所示。可以看到，模型的输入可以分为 System、User、Assistant 共3个部分，其中 System 和 User 部分是模型的 prompt 部分。prompt 中可以同时包含 text 和 audio（能读能听）。同理，Assistant 是模型生成的部分，也可以分别生成 text 和 audio（能写能说）。为了同时支持两种模态，模型配有2个 tokenizer，分别是 Text Tokenizer 和 Audio Tokenizer，可以将 text 和 audio 分别 token 化。

![higgs_audio_v2_architecture_combined](/Users/joywin/joywin/Technology-notes/week_report_2025/pics/higgs_audio_v2_architecture_combined.png)

Text 的 token 化跟普通的 LLM 是一样的。Audio 的 token 化在 2.2 中进行了介绍，1s 的音频经过  Audio Tokenizer 之后，会得到一个（1, 8, 25）的表征。由于这里加入了 RVQ 进行残差量化，因此，audio token 在进入模型之前，会把残差进行相加，最终 1s 的 audio 得到的就是一个 （1, 25）的表征，也就是 1s 共 25个 tokens。这样，audio token 就可以和 text  token 一起进入后面的模型，然后在同一个空间内进行计算。

## 3.1 DualFFN

与传统的 LLM 不同，Higgs Audio V2 需要同时处理 text 和 audio 两种模态的数据输入，因此，为了更好的处理不同的模态，Higgs Audio V2 在主体模型以 Llama-3.2-3B 为基座的基础上，同时进行了结构改进。将普通的 transformer block 变为了 dual FFN 结构。为了方便理解，将不同的 transformer block 变体做了以下的对比。

![dual_ffn](/Users/joywin/joywin/Technology-notes/week_report_2025/pics/dual_ffn.png)

首先是标准的 transformer block，主要由 attention 模块和 FFN 模块组成。MoE 结构则是对传统的 transformer block 进行了改进，通过将单个 FFN 结构拆分为多个子专家，从而可以降低计算成本，同时还有可能提升模型的效果。而 Higgs Audio V2 则是按照 text 和 audio  对 FFN 进行了拆分，不同的模态走不通的 FFN，也就是一个分模态的 MoE 结构。

模型的详细结构如下所示：

```python
HiggsAudioModel(
  (embed_tokens): Embedding(128256, 3072, padding_idx=128001)
  (audio_codebook_embeddings): Embedding(8208, 3072)
  (layers): ModuleList(
    (0-27): 28 x HiggsAudioDualFFNDecoderLayer(
      (self_attn): LlamaSdpaAttention(
        (q_proj): Linear(in_features=3072, out_features=3072, bias=False)
        (k_proj): Linear(in_features=3072, out_features=1024, bias=False)
        (v_proj): Linear(in_features=3072, out_features=1024, bias=False)
        (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (mlp): LlamaMLP(
        (gate_proj): Linear(in_features=3072, out_features=8192, bias=False)
        (up_proj): Linear(in_features=3072, out_features=8192, bias=False)
        (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
        (act_fn): SiLU()
      )
      (audio_mlp): LlamaMLP(
        (gate_proj): Linear(in_features=3072, out_features=8192, bias=False)
        (up_proj): Linear(in_features=3072, out_features=8192, bias=False)
        (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
        (act_fn): SiLU()
      )
      (audio_input_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
      (audio_post_attention_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
      (input_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
      (post_attention_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
    )
  )
  (norm): LlamaRMSNorm((3072,), eps=1e-05)
  (rotary_emb): LlamaRotaryEmbedding()
  (audio_decoder_proj): HiggsAudioDecoderProjector(
    (text_lm_head): Linear(in_features=3072, out_features=128256, bias=False)
    (audio_lm_head): Linear(in_features=3072, out_features=8208, bias=False)))
```

可以看到，text 有一个 128k 的词表，audio 的词表大小为 8192（8*1024）。一共有28层的 HiggsAudioDualFFNDecoderLayer，每一层又分别包含一个 attention 层，一个 mlp（text 支路） 和一个 audio_mlp （audio 支路）。这样的设计，可以使得 text 和 audio 的token 可以在同一个 attention 层中进行交互，但是在 FFN 中分别使用自己模态的单独参数，既完成了交互，又避免了共用同一个 FFN 可能带来的参数间互相冲突（妙啊）。输出层又分别包含一个 128k 的 text_lm_head 和一个 8192 的 audio_lm_head，从而分别对 text 和 audio token 进行生成预测。

这里的分模态 dual FFN 和之前的一些多模态的工作（比如 VLMO），有异曲同工之妙，都是将不同的模态，分别走不同的 FFN 的 MoE 结构。这里，由于每个 token 是属于 text 还是 audio 是明确的，并不需要一个路由模块决定每个 token 走哪个专家，只需要通过一个 mask 进行区别，将不同模态的 token 分别走不通的通路即可，代码如下：

```python
residual = hidden_states

text_hidden_states = self.post_attention_layernorm(hidden_states[~real_audio_out_mask])
audio_hidden_states = self.audio_post_attention_layernorm(hidden_states[real_audio_out_mask])

text_hidden_states = self.mlp(text_hidden_states)
residual[~real_audio_out_mask] += text_hidden_states

audio_hidden_states = self.audio_mlp(audio_hidden_states)
residual[real_audio_out_mask] += audio_hidden_states
```

Dual FFN 的消融结果如下，可以看到，使用 dual FFN 之后，基本上能有一个稳定的效果的提升。图中的 delay pattern 会在下一小节进行介绍。

![dual_ffn_ablation](/Users/joywin/joywin/Technology-notes/week_report_2025/pics/dual_ffn_ablation.png)

MoE 的思路真的是可以用在很多场景，通过将 FFN 模块进行拆分和隔离，避免了可能存在的参数冲突，从而提升模型效果。这里插一个小广告，如果想了解更多 deepseek 背后的 MoE 选择的逻辑，可以参考之前的这篇文章 [深度求索DeepSeek背后的底层逻辑](https://mp.weixin.qq.com/s/Wd-8IGqUFOJ1mEBvhL7cDw) 。

## 3.2 Delay pattern

在上一节的模型结构中，我们可以看到，audio_lm_head 的 out_features 是 8192，也就是表示模型会同时预测 8 个分残差的 audio tokens，这个和 text_lm_head 单次只预测一个 text token 是有不同的。这里就涉及到一个问题，后续残差层的 token 是和前面层的 token 有关联的，假设 $\hat{x}=q_1+q_2+\cdots+q_N$ ， $q_2$ 是 $\hat{x}$ 和 $q_1$ 的残差，那么 $q_2$ 的预测需要先知道 $q_1$ 才行。但如果每次只预测一个残差后的 token，那么整体预测的序列长度将会变为原始的 8 倍，计算量大幅增加。因此，为了在不增加序列长度的情况下，还能先预测  $q_1$ ，再预测  $q_2$ ，模型采用了delay pattern 的模式进行预测。 示意图如下：

![delay_pattern](/Users/joywin/joywin/Technology-notes/week_report_2025/pics/delay_pattern.png)

从图中可以看到，在时刻1，只预测 1st codebook 的第一个 token，在时刻2，预测 1st codebook 的第二个 token 和 2nd codebook 的第一个 token，依次类推。这样就同时预测多个token，并且，后续层的残差 token 也可以拿到前面层的前序信息。保证了信息量的同时，大大加速了推理过程。（妙啊，再一次～）

同时，由于 audio_lm_head 的维度是 8192，相当于不同的 codebook 之间没有共享参数，可以针对不同层的数据特点，分别训练得到自己的参数。

笔者最近在研究生成式推荐中的 tokenizer，Higgs Audio V2 中的双流 tokenizer 设计和 delay pattern ，都为生成式推荐的语义 ID 的设计及推理提供了新的优化思路。

# 4. 数据

数据对于模型的训练效果至关重要，那么 Higgs Audio V2 是如何构建数据的呢？

## 4.1 数据的构建

首先，由于我们想要生成的音频能够遵循复杂的指令，那么在训练数据中，首先要给出复杂的指令和对应的音频。复杂的指令，就需要包括需要生成音频的声学特征（场景环境等）、语义特征（对话的主题等）、以及每个人物说话的特点。这部分复杂的指令会放在 SYSTEM 的 prompt 中，然后在 USER 的 prompt 中，会提供给模型不同的对话文本，最终模型在 ASSISTANT 中，返回符合场景特点的、不同说话人的、对应文本的音频内容。这样的话，就只需要在 SYSTEM 中给出一次整体指令，然后在 USER 中不断给出人物对话的文本，模型就可以不断的生成对应的音频内容。

一个具体的数据样例如下所示：

```text
SYSTEM
    ACOUSTIC: This audio clip likely recorded indoors in a quiet environment...
    SEMANTIC: ...a dialogue between the host and her guest, a meditation teacher and author. They discuss...
    SPEAKER0: female, 30s-40s, clear and warm tone, ...
    SPEAKER1: male, clam and reflective tone, ...

USER
    [SPEAKER1] So we've got to look at ways to reducing that.
    [SPEAKER0] Yeah, there's a lot of us that, you know.

ASSISTANT
    [AUDIO]

USER
    [SPEAKER0] unless it's brought to our attention...

ASSISTANT
    [AUDIO]
```

那这里就会涉及到一个问题，我们可以从网络上爬取到很多的音频内容，但是这些音频内容中，并不会包含一个对应的音频的指令描述。所以需要对这些数据进行打标签，这其实是一个将 text 和 audio 进行对齐的工作。这个工作很麻烦，但又很重要。一种办法就是直接调用已有大模型的 API，对音频进行打标签，但这样可能会有法律风险（很多大模型公司都明令禁止使用 API 清洗数据，蒸馏新模型），而且成本也会非常高。所以，沐神他们采用的方法是，使用同样的模型架构，再训练一个语音理解模型。给定一个语音，请模型分析它的场景、人物设定，对话内容、人物情绪状态等等。相当于是把生成音频的任务反过来了，把上面的输入输出反过来，得到的就是音频理解模型的数据，举个例子如下：

```text
SYSTEM
		The user will provide you with an audio clip. Please analyze the scene of this audio, the characters involved, their personality traits, emotional states, and the content of their conversation.

USER
		[AUDIO]

ASSISTANT
		ACOUSTIC: This audio clip likely recorded indoors in a quiet environment...
    SEMANTIC: ...a dialogue between the host and her guest, a meditation teacher and author. They discuss...
    SPEAKER0: female, 30s-40s, clear and warm tone, ...
    SPEAKER1: male, clam and reflective tone, ...
    [SPEAKER1] So we've got to look at ways to reducing that.
    [SPEAKER0] Yeah, there's a lot of us that, you know.
```

这个就非常有意思，沐神举例子说，如果想教一个徒弟，让他会拳脚功夫，但是又没法一次教会，那解决办法就是一次教两个徒弟，第一个徒弟会打拳、第二个徒弟会踢腿， 然后让两个徒弟天天在一起互相对打，这样他们就会共同进步， 到最后希望他们两个都能学会对方的拳脚功夫。属实是有点妙啊。

## 4.2 数据量级 

通过左脚踩右脚，可以得到大量的有标签的数据。那么 Higgs Audio V2 的训练一共使用了多少数据量呢？沐神提到了，一共用了1000w小时的音频数据。我们来简单算下，$1000W\  hours = 10^7 * 3600 *25 = 9*10^{11} = 0.9 T\ tokens $ 。也就是说，在原始 Llama-3.2-3B 的基础上，又进一步增加了 0.9T audio tokens。这个量级已经非常多了，但是和在 text LLM 领域的 token 数相比，还有可以进一步 scaling 的空间。当然，数据要扩充起来是非常费力的一个工作，但通过数据的量级，我们可以知道当前 audio 大模型的发展阶段。目前，audioLLM 的 audio 数据量大概还是和 llama1（2023年的工作） 的数据量在同一个量级，未来还有很大的空间可以继续 scaling，也即是说，模型效果还可以变得更好，不禁让我对未来多模态端到端模型的效果有了进一步的期待。

| 任务      | 模型           | 数据量 |
| --------- | -------------- | ------ |
| Audio LLM | Higgs Audio V2 | 0.9T   |
| LLM       | llama1         | 1.4T   |
| LLM       | llama3         | 15T    |
| LLM       | deepseek V3    | 14.8T  |
| LLM       | Qwen3          | 36T    |



# 5. 模型效果

上面讲了这么多，那么模型实际的效果怎么样呢？有没有评测的结果呢？当然是有的，简单来说就是音频生成领域的 SOTA。接下来看看，是如何对 audio 生成结果进行评估的。

首先，是和 gpt-4o-mini-tts 的音频生成进行效果对比，看看谁的胜率更高。使用 Gemini 2.5 Pro 作为裁判，结果如下图所示。可以看到，在75.7%的情况下， Higgs Audio V2 都可以打赢  gpt-4o-mini-tts ，这个胜率也大幅领先其他的音频生成模型。

![performance](/Users/joywin/joywin/Technology-notes/week_report_2025/pics/performance.png)

除了直接打擂台，也有计算在音频生成 benchmark 上的各项指标。这里我们还是先介绍下音频生成评估中的常用指标，分别是 WER 和SIM。

- WER (Word Error Rate) ：WER 是衡量自动生成的语音被 ASR 转录成文本后，与原始目标文本之间差异的指标。具体而言是将 ASR 转录文本转换为目标文本所需进行的插入（Insertions）、删除（Deletions）和替换（Substitutions）操作的总次数，再除以目标文本中的总词数。公式： WER = (I + D + S) / N。一般用于衡量生成语音对输入文本内容的准确性。

- SIM:  SIM 衡量的是生成音频与参考音频的相似程度，一般会分别从生成音频和参考音频中提取 audio embeddings，并以这两个嵌入之间的余弦相似度作为 SIM 指标。一般用于衡量生成语音对参考音频说话人特征的忠实度。

在 Seed-TTS Eval 和 ESD 这两个测试集上，评估指标如下所示。可以看到，Higgs Audio V2 在这两个 benchmark 上 SIM 指标都达到了 SOTA。不过这里有个有意思的现象，就是 Higgs Audio V2 在 WER 上都不是最优，而且对比 Higgs Audio V1 和 V2，反而是 SIM 变好， WER 就会变差。这里盲猜两种可能性：1、音频在 ASR 过程中，ASR 模型有一定的误差，导致 WER 计算不准确；2、SIM 和 WER 这两个指标可能存在一定的互斥性，会此消彼长。如果有音频生成领域的大佬，可以在评论区讨论下。

![performance2](/Users/joywin/joywin/Technology-notes/week_report_2025/pics/performance2.png)

# 6. 总结

到此，整个工作就都介绍完了，我们总结一下。首先，我们了解了端到端的 audioLLM 模型的优势：端到端优化、复杂指令遵循、低延时。接下来，我们了解了 audio 如何表征、audio 的语义理解模型 hubert、以及 RVQ，并基于这些知识介绍了双流 tokenizer。再接着，我们了解了模型的 dual FFN 结构和解码过程的 delay pattern。最后，我们了解了数据构建过程和模型效果，并分析了接下来模型还有进一步 scaling 的可能性。

沐神在更新完第一个视频之后，也挖了一个坑说要讲讲模型细节，但应该是太忙了，一直没时间填坑。看完本文，如果觉得对你有点帮助，不妨点个赞，顶上去也让沐神看看，本文是不是能帮沐神填个坑～

![keng](/Users/joywin/joywin/Technology-notes/week_report_2025/pics/keng.png)



# 7. 参考资料

- https://github.com/boson-ai/higgs-audio
- https://www.bilibili.com/video/BV1LGbozkEDY/

