# LLM vocab

- vocab初始化

  ```python
  llama_vocab = llama_tokenizer.get_vocab()
  ziya_vocab = ziya_tokenizer.get_vocab()
  llama_rev_vocab = {v:k for k, v in llama_vocab.items()}
  ziya_rev_vocab = {v:k for k, v in ziya_vocab.items()}
  
  def generate_new_embeddings():
      new_embedding = torch.nn.Embedding(39424, 5120, 0).cuda()
      old_mebedding = llama.model.embed_tokens
      new_embedding.weight.data[:32000] = old_mebedding(torch.LongTensor(list(range(32000))).cuda())
      
      for i in range(32000, 39410):
          char = ziya_rev_vocab[i]
          input_ids = llama_tokenizer(char).input_ids[2:]
          target_id = ziya_tokenizer(char).input_ids[2:]
          if len(target_id) != 1:
              continue
  
          avg_embeddings = old_mebedding(torch.LongTensor(input_ids).cuda()).mean(axis=0)
          new_embedding.weight.data[target_id[0]] = avg_embeddings
      return new_embedding
  
  def generate_new_lm_head():
      return torch.nn.Linear(5120, 39424, bias=False)
  ```

  

