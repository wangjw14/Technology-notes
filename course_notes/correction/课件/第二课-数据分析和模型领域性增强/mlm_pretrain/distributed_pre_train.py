# coding utf-8
import os
import torch
from time import strftime, localtime
import random
import time
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import re
from utils import set_seed,ready_pretrain_data
from transformers import BertModel, BertConfig, BertForMaskedLM, AdamW
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer
from torch.utils.data import Dataset
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

#启动命令：CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 distributed_pre_train.py

# 单机多进程多卡分布式训练
# 1) 初始化
torch.distributed.init_process_group(backend="nccl")

# 2） 配置每个进程的gpu
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
nprocs = torch.cuda.device_count()

class bert_config:
    def __init__(self):
        # 数据路径
        self.data_dir = './data'
        self.bert_dir = './bert_data/raw_bert'
        self.bert_save_dir = './bert_data/pre_bert'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_train_epochs = 10  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.learning_rate = 2e-5  # 学习率
        self.model_name = None  # bert
        self.weight_decay = 0.01  # 权重衰减因子
        self.warmup_proportion = 0.1  # Proportion of training to perform linear learning rate warmup for.
        self.seed = 2022
        # 差分学习率
        self.diff_learning_rate = False
        # prob
        self.n_gpu = torch.cuda.device_count()
        self.vocab_size = None
        self.max_grad_norm = 1
        self.mode = 'ngram_mask'  # dynamic_mask,ngram_mask


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class BuildDataSet(Dataset):
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset
        self._len = len(self.dataset)
        self.vocab = list(set(re.findall('[a-zA-Z0-9\u4e00-\u9fa5]',''.join(dataset))))
        self.tokenizer = BertTokenizer(os.path.join(config.bert_dir,'vocab.txt'))

    def __getitem__(self, index):
        example = self.dataset[index]
        if config.mode == "ngram_mask":
            example, label = self.ngram_mask(example)
        else:
            example, label = self.dynamic_mask(example)

        assert len(example) == len(label)
        return (example, label)

    def __len__(self):
        return self._len

    def ngram_mask(self, text_ids):
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        idx = 0
        while idx < len(rands):
            if rands[idx] < 0.15:  # 需要mask
                ngram = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])  # 若要mask，进行n_gram mask的概率
                if ngram == 3 and len(rands) < 7:  # 太大的gram不要应用于过短文本
                    ngram = 2
                if ngram == 2 and len(rands) < 4:
                    ngram = 1
                L = idx + 1     #idx位置为确定性mask，所以+1
                R = idx + ngram  # 最终需要mask的右边界（开）
                while L < R and L < len(rands):
                    rands[L] = np.random.random() * 0.15  # 强制mask
                    L += 1
                idx = R
                if idx < len(rands):
                    rands[idx] = 1  # 禁止mask片段的下一个token被mask，防止一大片连续mask
            idx += 1

        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8:
                input_ids.append('[MASK]')
                output_ids.append(i)  # mask预测自己
            elif r < 0.15 * 0.9:
                input_ids.append(i)
                output_ids.append(i)  # 自己预测自己
            elif r < 0.15:
                input_ids.append(self.vocab[np.random.choice(len(self.vocab))])
                output_ids.append(i)  # 随机的一个词预测自己，随机词不会从特殊符号中选取，有小概率抽到自己
            else:
                input_ids.append(i)
                output_ids.append('[PAD]')  # 保持原样不预测
        return input_ids, output_ids

    def dynamic_mask(self, text_ids):
        """dynamic mask
        """
        input, output = [], []
        rands = np.random.random(len(text_ids))
        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8:
                input.append('[MASK]')
                output.append(i)    # mask预测自己
            elif r < 0.15 * 0.9:
                input.append(i)     # 自己预测自己
                output.append(i)
            elif r < 0.15:
                input.append(self.vacab[np.random.choice(len(self.vocab))])
                output.append(i)        # 随机的一个词预测自己，随机词不会从特殊符号中选取，有小概率抽到自己
            else:
                input.append(i)
                output.append('[PAD]')
        return input, output


    def collate_fn(self,batch_data):
        max_len = max([len(x[0]) for x in batch_data]) + 2
        input_ids, token_type_ids, attention_mask, labels = [], [], [], []
        for text, label in batch_data:
            inputs = self.tokenizer.encode_plus(text=text,
                                           max_length=max_len,
                                           padding='max_length',
                                           is_pretokenized=True,
                                           return_token_type_ids=True,
                                           return_attention_mask=True, truncation=True)
            label = self.tokenizer.encode_plus(text=label,
                                          max_length=max_len,
                                          padding='max_length',
                                          is_pretokenized=True,
                                          return_token_type_ids=False,
                                          return_attention_mask=False, truncation=True)
            input_ids.append(inputs['input_ids'])
            token_type_ids.append(inputs['token_type_ids'])
            attention_mask.append(inputs['attention_mask'])
            labels.append(label['input_ids'])
        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).float()
        labels = torch.tensor(labels).long()
        return input_ids, token_type_ids, attention_mask, labels
    

def pre_trained(config):
    train = ready_pretrain_data(config.data_dir)

    logger.info("pretrain data nums:{}".format(len(train)))

    train_dataset = BuildDataSet(train, config)
    # 3）使用DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_load = DataLoader(dataset=train_dataset, batch_size=config.batch_size,sampler=train_sampler,
                            shuffle=False, collate_fn=train_dataset.collate_fn,num_workers=6)

    # load source bert weight
    model = BertForMaskedLM.from_pretrained(config.bert_dir, output_hidden_states=True, hidden_dropout_prob=0.1)

    for param in model.parameters():
        param.requires_grad = True

    # 4) 封装之前要把模型移到对应的gpu
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    cudnn.benchmark = True
    model.train()
    for epoch in range(config.num_train_epochs):
        train_sampler.set_epoch(epoch)
        torch.cuda.empty_cache()
        total_loss = 0
        for batch, (input_ids, token_type_ids, attention_mask, label) in enumerate(tqdm(train_load)):
            input_ids = input_ids.cuda(local_rank, non_blocking=True)
            attention_mask = attention_mask.cuda(local_rank, non_blocking=True)
            token_type_ids = token_type_ids.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)
            loss = model(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids, labels=label)[0]

            # 同步各个进程的速度,计算分布式loss
            torch.distributed.barrier()
            if torch.cuda.device_count() > 1:
              reduced_loss = reduce_mean(loss,nprocs)
            else:
              reduced_loss = loss

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            total_loss+=reduced_loss
            if batch % 100:
                torch.cuda.empty_cache()

        if local_rank == 0:
            avg_loss = total_loss / len(train_load)
            now = strftime("%Y-%m-%d %H:%M:%S", localtime())
            logger.info("time:{},epoch:{}/{},avg_loss:{}".format(now, epoch + 1, config.num_train_epochs, avg_loss.item()))
            checkpoint = model.module.state_dict(),
            torch.save(checkpoint,os.path.join(config.bert_save_dir,"pytorch_model.bin"))
            del checkpoint


if __name__ == '__main__':
    
    logger.info("train start：{}".format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))
    config = bert_config()
    set_seed(config.seed)
    pre_trained(config)
    logger.info("train finished：{}".format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))

