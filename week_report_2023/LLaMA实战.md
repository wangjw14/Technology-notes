## LLaMA实战

- CUDA OOM during model saving.
  - Assume you are using torch=1.13.0, change python/lib/python3.9/site packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:2224 from state_dict[fqn] = state_dict[fqn].clone().detach() to state_dict[fqn] = state_dict[fqn].cpu().clone().detach()
  - https://github.com/tatsu-lab/stanford_alpaca/issues/81

- apex安装
  - 注意cuda版本，需要和torch的cuda编译版本一致。一般是cuda11.7
- 