

```sh

data_list="whole_artile"
model_list="Qwen-LLaMAfied-HFTok-7B-Chat"
for model in $model_list;
    do
    for data in $data_list;
    do
        /root/paddlejob/workspace/env_run/env/python/bin/deepspeed --num_gpus=8 --master_port=20001 fastchat/train/train_mem.py \
            --model_name_or_path /root/paddlejob/workspace/env_run/wjw/models/$model  \
            --data_path /root/paddlejob/workspace/env_run/wjw/FastChat/data/$data.json \
            --eval_data_path /root/paddlejob/workspace/env_run/wjw/FastChat/data/eval.json \
            --bf16 True \
            --output_dir /root/paddlejob/workspace/env_run/output/model/${model}_${data} \
            --num_train_epochs 1 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 2 \
            --evaluation_strategy "steps" \
            --eval_steps 10 \
            --save_strategy "steps" \
            --save_steps 1500 \
            --save_total_limit 8 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.04 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --deepspeed /root/paddlejob/workspace/env_run/wjw/FastChat/scripts/configs/deepspeed_config.json\
            --tf32 True \
            --model_max_length 4096 \
            --gradient_checkpointing True \
            --lazy_preprocess True \
            --template "whole_text" \
            --report_to "tensorboard" \
            --logging_dir /root/paddlejob/workspace/env_run/output/tensorboard/${model}_${data}
    done
done


/root/paddlejob/workspace/env_run/env/python/bin/python3 /root/paddlejob/workspace/env_run/wjw/utils/run.py


```

