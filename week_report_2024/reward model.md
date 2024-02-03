# reward model

- ppo理论
- ppo代码
- reward model
- 数据



- 多模态
- moe





```
AutoModelForCausalLMWithValueHead(
  (pretrained_model): LlamaForCausalLM(
    (model): LlamaModel(
      (embed_tokens): Embedding(151936, 4096, padding_idx=0)
      (layers): ModuleList(
        (0-31): 32 x LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (norm): LlamaRMSNorm()
    )
    (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
  )
  (v_head): ValueHead(
    (dropout): Dropout(p=0.1, inplace=False)
    (summary): Linear(in_features=4096, out_features=1, bias=True)
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
)
```

