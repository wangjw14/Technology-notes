

import json
import pandas as pd
import transformers
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

path = '/root/paddlejob/workspace/env_run/LLaMA-Factory/2024Q1base.xlsx'

df = pd.read_excel(path)
def ocr_2_subtitle(row):
    ocr = json.loads(row['ocr'])
    ocr = "".join([f"{o[0]}:{o[1]}" for o in ocr])
    return f"标题：{row['title']}。正文：{ocr}"

def t_2_int(t):
    m, s = t.split(":")
    return 60*int(m) + int(s)

def parse(x):
    # ['0:00:审美追求', '0:53:建盏特色', '1:54:乌金釉的象征', '3:04:乌金釉的稀有', '3:27:建盏的影响力']
    x = eval(x)
    time = [t_2_int(i[:4]) for i in x]
    text = [i[5:] for i in x]
    res = []
    for ti, te in zip(time, text):
        res.append(f"{ti}:{te}")
    return "。".join(res)

df['subtitle'] = df.apply(lambda row: ocr_2_subtitle(row), axis=1)
df['res'] = df['节点v3离线-竞文'].apply(lambda x: parse(x))

print(df[['subtitle', 'res']].head())
# data_new = []
# with open(path, 'r') as f:
#     data = json.load(f)

# for d in data:
#     new_d = {}
#     new_d["loc"] = d['id']
#     new_d['subtitle'] = d['conversations'][0]['value']
#     new_d['gpt4'] = d['conversations'][1]['value']
#     data_new.append(new_d)

# df = pd.DataFrame(data_new)
# df.drop_duplicates(subset='loc', inplace=True)

# model_list = ['dpo_Qwen-llama-7B_final_bad_1w_filter_full_epoch1_lr5e-6', 'dpo_Qwen-llama-7B_final_bad_1w_filter_full_epoch3_lr5e-6', 'dpo_Qwen-llama-7B_good_bad_1w_full_epoch1_lr5e-6', 'dpo_Qwen-llama-7B_good_bad_1w_full_epoch3_lr5e-6', 'Qwen-llama-7B_summary_14k_correct_0_full_epoch1_lr5e-6']

model_list = ['rm_Qwen-llama-7B_final_bad_1w_filter_full_epoch1_lr5e-6', 
                'rm_Qwen-llama-7B_final_bad_1w_filter_full_epoch3_lr5e-6']

from typing import TYPE_CHECKING, Optional, Tuple, Dict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from transformers.utils import cached_file
import torch, os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

SAFE_WEIGHTS_NAME = "model.safetensors"
if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def load_valuehead_params(path_or_repo_id: str, model_args: "ModelArguments") -> Dict[str, torch.Tensor]:
    kwargs = {
        "path_or_repo_id": path_or_repo_id,
    }

    try:
        from safetensors import safe_open
        vhead_file = cached_file(filename=SAFE_WEIGHTS_NAME, **kwargs)
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            return {
                "v_head.summary.weight": f.get_tensor("v_head.summary.weight"),
                "v_head.summary.bias": f.get_tensor("v_head.summary.bias")
            }
    except Exception as err:
        print("Failed to load {}: {}".format(SAFE_WEIGHTS_NAME, str(err)))
    return None


def load_model_and_tokenizer(
    model_name_or_path,
    finetuning_args: "FinetuningArguments",
    is_trainable: Optional[bool] = False,
    add_valuehead: Optional[bool] = False
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    config_kwargs = {
        "trust_remote_code": True,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        split_special_tokens=False,
        padding_side="right", # training with left-padded tensors in fp16 precision may cause overflow
        **config_kwargs
    )
    config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        **config_kwargs
    )

    if add_valuehead:
        model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        vhead_path = model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, {})
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            print("Loaded valuehead from checkpoint: {}".format(vhead_path))


    model.requires_grad_(False) # fix all model params
    model.eval()
    model.to("cuda")
    
    return model, tokenizer




def infer(model, tokenizer, row):
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman:视频信息如下，请对其进行分段和总结。{row['subtitle']}。视频信息结束，开始分段总结。{row['res']}\nAssistant:"
    # try:
    inputs = tokenizer([prompt])
    inputs = {k: torch.tensor(v).to("cuda") for k, v in inputs.items()}
    length = list(inputs['input_ids'].size())
    o1,o2, values = model(**inputs, output_hidden_states=True, return_dict=True)
    # print(values)
    # print(length, values.shape)
    # output_ids = model.generate(
    #     **inputs,
    #     do_sample=False,
    #     num_beams=1,
    #     repetition_penalty=1.0,
    #     max_new_tokens=512,)
    # out = tokenizer.decode(output_ids[0])
    # out = out.split('Assistant:')[1]
    values = values.cpu().tolist()
    # print(length, values[-1][-1], values)
    return length, values[-1][-1]
    # except:
    #     return ""



for model_name in model_list:
    model_name_or_path = "/root/paddlejob/workspace/env_run/output/model/" + model_name
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, {}, add_valuehead=True)
    print(model)
    df[model_name] = df.apply(lambda row: infer(model, tokenizer, row), axis=1)
    del model 
    del tokenizer


df.to_excel("/root/paddlejob/workspace/env_run/LLaMA-Factory/eval/Q1_eval_data_with_rm.xlsx", index=None)
