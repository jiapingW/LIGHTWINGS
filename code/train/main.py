import argparse
import deepspeed
import os
from forked_pdb import ForkedPdb
import os, json, hashlib, pickle

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='')

parser.add_argument('--trainpath', type=str, default="qwen2_data/sft_data/all.jsonl")
parser.add_argument('--testpath', type=str,
                    default=None)
parser.add_argument('--savedir', type=str, default='/ckpt')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument("--choose_model",type=str,default='llama',help='choose the model you want to train')
parser.add_argument("--max_length",type=int,default=4096,help="max length of input")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
# print('args:',args)
import json
import re


deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)
if args.choose_model == 'llama':
    config_path = 'configs/llama_config.json'
elif args.choose_model == 'qwen' or args.choose_model == 'qwen_linear' or args.choose_model == 'qwen_linear_new' or args.choose_model == 'qwen_linear_new_new' or args.choose_model == 'qwen_performer':
    config_path = 'configs/qwen_config.json'
elif args.choose_model == 'qwen_code':
    config_path = 'configs/qwen_code_config.json'
elif 'qwen3' in args.choose_model:
    config_path = 'configs/qwen3_config.json'
else:
    raise Exception('choose model error')


train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": 20, # 默认 40
    "num_workers": 4,
    "max_len": 4096,
    # "config_path": "llama_config.json" if args.choose_model=='llama' or args.choose_model=='llama_linear' else "qwen_config.json"
    "config_path": config_path
}

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from llama import padding

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate.utils import set_seed

set_seed(0)
# from cnets import Model
from configs import EConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
# import accelerate
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup
from utils import preprocess_conversations,_make_cache_key 
from template import TEMPLATE_REGISTRY


def build_dataset_rank(
        tokenizer, datapath
):

    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds
    original_columns1 = ds1.column_names
    num_proc = 8 # 默认8

    
    def preprocess_function(examples):
        # ForkedPdb().set_trace()
        if args.choose_model == 'llama' or args.choose_model == 'llama_linear':
            return preprocess_conversations(tokenizer, examples, TEMPLATE_REGISTRY.templates['llama3'], args.max_length, args.trainpath)
        elif args.choose_model == 'qwen' or args.choose_model == 'qwen_linear' or args.choose_model == 'qwen_linear_new' or args.choose_model == 'qwen_linear_new_new' or args.choose_model == 'qwen_performer':
            return preprocess_conversations(tokenizer, examples, TEMPLATE_REGISTRY.templates['qwen2'], args.max_length, args.trainpath)
        elif args.choose_model == 'qwen_code' or 'qwen3' in args.choose_model: # qwen3 用这个system propmt
            return preprocess_conversations(tokenizer, examples, TEMPLATE_REGISTRY.templates['qwen'], args.max_length, args.trainpath)
        else:
            raise Exception('unsupported model')

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )


    ds1.set_format(type="torch")
    return ds1


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset_rank(tokenizer, args.trainpath)
if args.testpath:
    testdataset = build_dataset_rank(tokenizer, args.testpath)

# LlamaConfig
config = EConfig.from_pretrained(train_config["config_path"]) 

if args.choose_model=='llama':
    from llama import Model
    model = Model(config, path=args.basepath, load_emb=True, load_head=True)
elif args.choose_model=='qwen' or args.choose_model=='qwen3':
    from qwen import Model
    model = Model(config, path=args.basepath, load_emb=True, load_head=True)
elif args.choose_model=='qwen_performer':
    from qwen_performer import Model
    model = Model(config, path=args.basepath, load_emb=True, load_head=True)
elif args.choose_model=='qwen_linear_new':
    from qwen_linear_new import Model
    model = Model(config, path=args.basepath, load_emb=True, load_head=True)
elif args.choose_model=='qwen_linear_new_new' or args.choose_model=='qwen3_linear_new_new':
    from qwen_linear_new_new import Model
    model = Model(config, path=args.basepath, load_emb=True, load_head=True)
elif args.choose_model=='qwen_code':
    from qwen_code import Model
    model = Model(config, path=args.basepath, load_emb=True, load_head=True)
else:
    raise Exception('不支持的模型')


model.scandata(args.trainpath, args.basepath, args.max_length)


criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]

model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     )
global_step = 0
client_state = {}


global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()
if global_rank == 0:
    writer = SummaryWriter(log_dir=os.path.join(args.savedir, "runs"),
                           purge_step=global_step)        # 继续 step 计数


os.makedirs(args.savedir, exist_ok=True)
if args.testpath:
    sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, num_workers=4, pin_memory=True,
                            collate_fn=DataCollatorWithPadding())

train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler, num_workers=4,
                          pin_memory=True,
                          collate_fn=DataCollatorWithPadding())


def find_max_state_with_file(directory, filename="zero_to_fp32.py"):
    max_a = -1
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_a = max(max_a, a_value)
    if max_a == -1:
        return None, 0
    return f"{directory}/state_{max_a}", max_a + 1


checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)

if checkpoint_path:
    model_engine.load_checkpoint(checkpoint_path)
    # global_step = client_state.get("global_step", 0)
    global_step=29650
    if global_rank == 0:
        print(f"Resume from {checkpoint_path}, epoch={start_epoch}, step={global_step}")

for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch+1)
    print(f"Now training epoch {epoch}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]


    for batch_idx, data in enumerate(tqdm(train_loader)):
        model.zero_grad()
        # print(f'input_ids:{data["input_ids"].shape}, attention_mask:{data["attention_mask"].shape}')
        plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                               attention_mask=data["attention_mask"].to(rank),
                                               loss_mask=data["loss_mask"],
                                               )

        ploss_weight = [0.8 ** i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        loss = ploss
        model_engine.backward(loss)

        model_engine.step()
        global_step += 1  

        if global_rank == 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            for key, value in logdict.items():
                writer.add_scalar(key, value, global_step)
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]


    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            writer.add_scalar(f"train/epochacc_{i}", acc_i, epoch)
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            # wandb.log({f"train/epochploss_{i}": loss_i})
            writer.add_scalar(f"train/epochploss_{i}", loss_i, epoch)
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")

    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    if args.testpath:
        for batch_idx, data in enumerate(tqdm(test_loader)):
        
            with torch.no_grad():
                plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                                    attention_mask=data["attention_mask"].to(rank),
                                                    loss_mask=data["loss_mask"],
                                                    )
                epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
                epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]
        
        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
            acc_i = acc_i.item()
            if global_rank == 0:
                writer.add_scalar(f"test/epochacc_{i}", acc_i, epoch)
                print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
            loss_i = loss_i.item()
            if global_rank == 0:
                writer.add_scalar(f"test/epochploss_{i}", loss_i, epoch)
                print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")


    model_engine.save_16bit_model(f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True)
    if epoch % 1 == 0:
        client_state = {"global_step": global_step}
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.savedir}/state_{epoch}", client_state=client_state)


if global_rank == 0:
    writer.close()