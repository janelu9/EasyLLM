#!/usr/bin/env python
# coding: utf-8
# Created on Thur Jun 29 09:36:49 2023
# @author: Lu Jian
# Email:janelu@live.cn;

import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)
    
def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      fast_tokenizer=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                  fast_tokenizer=True)
    return tokenizer


def save_hf_format(model, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

def get_param_groups(module,
                     no_weight_decay_cond=None,
                     scale_lr_cond=None,
                     lr_mult=1.0):
    """creates param groups based on weight decay condition (regularized vs non regularized)
       and learning rate scale condition (args.lr vs lr_mult * args.lr)
       scale_lr_cond is used during finetuning where head of the network requires a scaled
       version of the base learning rate. 
    """
    wd_no_scale_lr = []
    wd_scale_lr = []
    no_wd_no_scale_lr = []
    no_wd_scale_lr = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        if no_weight_decay_cond is not None:
            no_wd = no_weight_decay_cond(name, param)
        else:
            # do not regularize biases nor Norm parameters
            no_wd = name.endswith(".bias") or len(param.shape) == 1

        if scale_lr_cond is not None:
            scale_lr = scale_lr_cond(name, param)
        else:
            scale_lr = False

        if not no_wd and not scale_lr:
            wd_no_scale_lr.append(param)
        elif not no_wd and scale_lr:
            wd_scale_lr.append(param)
        elif no_wd and not scale_lr:
            no_wd_no_scale_lr.append(param)
        else:
            no_wd_scale_lr.append(param)

    param_groups = []
    if len(wd_no_scale_lr):
        param_groups.append({'name': 'wd_no_scale_lr', 'params': wd_no_scale_lr, 'wd_mult': 1.0, 'lr_mult': 1.0})
    if len(wd_scale_lr):
        param_groups.append({'name': 'wd_scale_lr', 'params': wd_scale_lr, 'wd_mult': 1.0, 'lr_mult': lr_mult})
    if len(no_wd_no_scale_lr):
        param_groups.append({'name': 'no_wd_no_scale_lr', 'params': no_wd_no_scale_lr, 'wd_mult': 0.0, 'lr_mult': 1.0})
    if len(no_wd_scale_lr):
        param_groups.append({'name': 'no_wd_scale_lr', 'params': no_wd_scale_lr, 'wd_mult': 0.0, 'lr_mult': lr_mult})

    return param_groups

def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict
