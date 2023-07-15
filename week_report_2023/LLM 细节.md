# LLM 细节

- 如何加速，vllm
- 多语言，压缩词表
- webGPT，webGLM
- chatlaw
- ACL，检索增强对话
- nbce
- 解码
- 
- sparse attention
- Transformers 代码及细节
  - https://github.com/wenjtop/transformer/tree/main
  - https://zhuanlan.zhihu.com/p/624740065

- 读代码
  - 

- 左padding还是右padding

- model.eval()
- flashattention





## 已完成

- pagedattention

- Llama 网络结构

  ```python
  LlamaForCausalLM(
    (model): LlamaModel(
      (embed_tokens): Embedding(32000, 5120, padding_idx=0)
      (layers): ModuleList(
        (0-39): 40 x LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (k_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (v_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=5120, out_features=13824, bias=False)
            (down_proj): Linear(in_features=13824, out_features=5120, bias=False)
            (up_proj): Linear(in_features=5120, out_features=13824, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (norm): LlamaRMSNorm()
    )
    (lm_head): Linear(in_features=5120, out_features=32000, bias=False)
  )
  
  model.layers.0.input_layernorm.weight torch.Size([5120])
  model.layers.0.post_attention_layernorm.weight torch.Size([5120])
  model.norm.weight torch.Size([5120])
  ```

  