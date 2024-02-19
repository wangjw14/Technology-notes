# reward model

- ppo理论
- ppo代码
- reward model
- 数据



- 多模态
- moe



|      | loss                                                         |
| ---- | ------------------------------------------------------------ |
| RM   | $\max _{r_\phi}\left\{\mathbb{E}_{\left(x, y_{\text {win }}, y_{\text {lose }}\right) \sim \mathcal{D}}\left[\log \sigma\left(r_\phi\left(x, y_{\text {win }}\right)-r_\phi\left(x, y_{\text {lose }}\right)\right)\right]\right\}$ |
| PPO  | $\max _{\pi_\theta}\left\{\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y \mid x)}\left[r_\phi(x, y)\right]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi_\theta(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right]\right\}$ |
| DPO  | $\max _{\pi_\theta}\left\{\mathbb{E}_{\left(x, y_{\text {win }}, y_{\text {lose }}\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta\left(y_{\text {win }} \mid x\right)}{\pi_{\text {ref }}\left(y_{\text {win }} \mid x\right)}-\beta \log \frac{\pi_\theta\left(y_{\text {lose }} \mid x\right)}{\pi_{\text {ref }}\left(y_{\text {lose }} \mid x\right)}\right)\right]\right\}$ |





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

