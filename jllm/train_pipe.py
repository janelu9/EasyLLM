#!/usr/bin/env python
# coding: utf-8
# Created on Thur Jun 29 09:36:49 2023
# @author: Jian Lu
# Email:janelu@live.cn;

import torch
import deepspeed
from transformers import (
    AutoConfig,
    SchedulerType,
    get_scheduler,)
from .utils import (
    dynamic_import_module,
    set_random_seed,
    get_param_groups,
    get_optimizer_grouped_parameters)
from .model import (
    autopartition_transformer,
    autopartition_decoder,
    get_virtual_num_hidden_layers,
    convert_linear_layer_to_lora,
    only_optimize_lora_parameters,
    make_model_gradient_checkpointing_compatible,
    ModelPipe,
    sequence_parallel as spu
    )
from .trainer import train
from .ds_config import get_train_ds_config
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.pipe import ProcessTopology
from functools import partial
import pyarrow.parquet
import json
import numpy as np
import os
import datetime
import argparse
import sys
os.environ['PATH']+=":"+os.path.dirname(sys.executable)
os.environ['TOKENIZERS_PARALLELISM']='true'

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')        
# Include DeepSpeed configuration arguments
parser.add_argument("--model",
                    type=str,
                    default= "baichuan-inc/Baichuan-13B-Chat",
                    help="huggingface's model path")
parser.add_argument("--train_data",
                    type=str,
                    default= "",
                    help="data for training,jsonl or parqet folder")
parser.add_argument("--eval_data",
                    type=str,
                    default= "",
                    help="data for evalution,jsonl or parqet folder")
parser.add_argument("--img_shm",
                    type=str,
                    default= None,
                    help="image files' shm dir if reading during traing")
parser.add_argument("--from_ckpt",
                    type=str,
                    default= "",
                    help="ckpt dir to load pretrained model parameters")
parser.add_argument("--resume_ckpt",
                    type=str,
                    default= "",
                    help="ckpt dir to resume interruption")
parser.add_argument("--only_load_model",
                    action='store_true',
                    help="only load model from checkpoint")
parser.add_argument('--tag',
                    type=str,
                    default=None,
                    help='checkpoint tag')
parser.add_argument("--ds_config",
                    type=str,
                    default= None,
                    help="deepspeed's config pyfile")
parser.add_argument('--zero_stage',
                    type=int,
                    default=0,
                    help='zero stage')
parser.add_argument('--split_dlayer',
                    action='store_true',
                    help='split decoder layers')
# parser.add_argument('--sort_by_len',
                    # action='store_true',
                    # help='sort the samples by length')
parser.add_argument('--moe_layer_pipe_size',
                    type=int,
                    default=2,
                    help='num layers inner one decoder layer')
parser.add_argument('--emb_partitions',
                    type=int,
                    default=1,
                    help='split embedding')
parser.add_argument('--timeout',
                    type=int,
                    default=1800,
                    help='timeout')
parser.add_argument('--pipe_parallel_size',
                    type=int,
                    default=1,
                    help='pipe parallel size')
parser.add_argument('--encoder_pipe_parallel_size',
                    type=int,
                    default=0,
                    help="encoder's pipe parallel size")
parser.add_argument('--tensor_parallel_size',
                    type=int,
                    default=1,
                    help='model parallel size')
parser.add_argument('--expert_parallel_size',
                    type=int,
                    default=1,
                    help='expert model parallel size')
parser.add_argument('--sequence_parallel_size',
                    type=int,
                    default=1,
                    help='sequence parallel size')
parser.add_argument('--attention_alpha',
                    type=float,
                    default=3,
                    help='coefficient to estimate attention\'s computation.')
parser.add_argument("--partition_method",
                    type=str,
                    default= "fast",
                    help="support 'fast', 'mem' and deepspeed's ")
parser.add_argument("--multi_layerspec",
                    action='store_true',
                    help='multi layers per stage')
parser.add_argument('--num_train_epochs',
                    type=int,
                    default=1,
                    help='train epochs')
parser.add_argument('--micro_batch_size',
                    type=int,
                    default=1,
                    help='per device train batch_size')
parser.add_argument('--global_batch_size',
                    type=int,
                    default=None,
                    help='global batch size')
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help='gradient accumulation steps')
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.,
                    help="Weight decay to use.")
parser.add_argument("--lr_scheduler_type",
                    type=SchedulerType,
                    default="cosine",
                    help="The scheduler type to use.",
                    choices=[
                        "linear", "cosine", "cosine_with_restarts", "polynomial",
                        "constant", "constant_with_warmup"
                    ])
parser.add_argument("--num_warmup_steps",
                    type=float,
                    default=0,
                    help="Number or rate of steps for the warmup in the lr scheduler.")
parser.add_argument("--learning_rate",
                    type=float,
                    default=3e-4,
                    help= "Initial learning rate (after the potential warmup period) to use.",)
parser.add_argument('--seq_len',
                    type=int,
                    default=2048,
                    help='max seq len')
parser.add_argument('--skip_train_steps',
                    type=str,
                    default="",
                    help='Do not train on these steps')
parser.add_argument('--steps_per_print',
                    type=int,
                    default=1,
                    help='steps per print')
parser.add_argument('--num_workers',
                    type=int,
                    default=None,
                    help='dataloader\'s num_workers')
parser.add_argument('--steps_per_eval',
                    type=int,
                    default=-1,
                    help='steps per eval')
parser.add_argument('--steps_per_checkpoint',
                    type=int,
                    default=-1,
                    help='steps per checkpoint')
parser.add_argument("--checkpoint",
                    type=str,
                    default= "",
                    help="checkpoint dir")
parser.add_argument('--background_executor',
                    type=str,
                    default='none',
                    choices=["process", "thread", "null", "none",""],
                    help='excutor of background')
parser.add_argument('--ckpt_step_gt',
                    type=int,
                    default=0,
                    help='checkpoint steps >= ckpt_step_gt')
parser.add_argument('--best_of',
                    type=int,
                    default=1,
                    help='checkpoint top k of eval_loss')
parser.add_argument('--ckpt_epoch',
                    type=str,
                    default="",
                    help='checkpoint the given epoches')
parser.add_argument('--skip_epoch',
                    type=str,
                    default="",
                    help='checkpoint except the given epoches')
parser.add_argument('--max_num_checkpoints',
                    type=int,
                    default=-1,
                    help='max checkpoint num')
parser.add_argument('--only_ckpt_model',
                    action='store_true',
                    help='Only checkpoint the model parameters.')
parser.add_argument('--only_ckpt_lora',
                    action='store_true',
                    help='Only checkpoint the lora parameters.')
parser.add_argument('--only_cache_model',
                    action='store_true',
                    help='Only cache the model.')
parser.add_argument('--force_4k',
                    action='store_true',
                    help='force the input_ids to 4k')
parser.add_argument('--early_stop',
                    type=int,
                    default=-1,
                    help='if eval loss continuous rebound epoches == early_stop, training will be breaked')              
parser.add_argument('--checkpoint_grad_interval',
                    type=int,
                    default=0,
                    help='The granularity activation checkpointing in terms of number of layers. 0 disables activation checkpointing.')
parser.add_argument('--no_checkpoint_grad_step',
                    type=int,
                    default=1000000,
                    help='no checkpoint grad step')
parser.add_argument('--no_shuf',
                    action='store_true',
                    help='disable shuffle at every epoch.')
parser.add_argument('--dist_data',
                    action='store_true',
                    help='distribute data on every node.')
parser.add_argument('--no_safetensor',
                    action='store_true',
                    help='not use safetensor.')
parser.add_argument('--async_tensor_model_parallel_allreduce',
                    action='store_true',
                    help='async tensor_model_parallel_allreduce')
parser.add_argument('--no_pin_memory',
                    action='store_true',
                    help='not pin memory ')
parser.add_argument('--init',
                    action='store_true',
                    help='train from 0')
parser.add_argument('--block_mask',
                    action='store_true',
                    help="use BlockDiagonalCausalMask")
parser.add_argument('--cache_model',
                    type=str,
                    default=None,
                    help='cached model dir')
parser.add_argument("--aux_loss_step",
                    type=float,
                    default=0.0,
                    help="moe aux loss step rat of train steps.")
parser.add_argument("--aux_loss_alpha",
                    type=float,
                    default=0.0,
                    help="moe aux loss alpha.")
parser.add_argument("--aux_loss_backward_scale",
                    type=float,
                    default=1.0,
                    help="aux loss backward scale.")
parser.add_argument("--swap_experts_per_steps",
                    type=int,
                    default=0,
                    help="swap experts per train steps.")
# parser.add_argument('--sequence_parallel',
                    # action='store_true',
                    # help='if enable sequence parallel.')
parser.add_argument('--balance_experts_timeout',
                    type=int,
                    default=60,
                    help="balance experts timeout.")
parser.add_argument('--pad_one_per_batch',
                    action='store_true',
                    help='pad the one samples every batch.')
parser.add_argument("--padding_rate",
                    type=float,
                    default=0.0,
                    help="padding the img length to padding_rate*max_num_patches.")               
parser.add_argument("--seed",
                    type=int,
                    default=1234,
                    help="A seed for reproducible training.")
parser.add_argument("--output_dir",
                    type=str,
                    default="",
                    help="Where to store the model.")
## LoRA for efficient training setting
parser.add_argument("--lora_dim",
                    type=int,
                    default=0,
                    help="If > 0, use LoRA for efficient training.")
parser.add_argument("--lora_alpha",
                    type=int,
                    default=1,
                    help="lora_alpha/lora_dim is the scaling.")
parser.add_argument("--lora_module_name",
                    type=str,
                    default= "qkv_proj,o_proj,gate_up_proj,down_proj",
                    help="The scope of LoRA.")
parser.add_argument('--only_optimize_lora',
                    action='store_true',
                    help='Only optimize the LoRA parameters.')
# rlhf
parser.add_argument('--rlhf',
                    action='store_true',
                    help='Reinforcement Learning.')
parser.add_argument('--isolated_vllm',
                    action='store_true',
                    help='isolated vllm')
parser.add_argument("--num_generations",
                    type=int,
                    default=4,
                    help="num generations")
parser.add_argument("--max_new_tokens",
                    type=int,
                    default=33,
                    help="max new tokens")
parser.add_argument("--vllm_sync_stage",
                    type=int,
                    default=0,
                    choices=(0,1,2,3,4),
                    help="how to sync weights to vllm. 0:ipc,1:nccl,2:cpu,3:nccl&ipc,4:nccl&cpu.")
parser.add_argument("--ray_ip",
                    type=str,
                    default=None,
                    help="ray's master ip.")
parser.add_argument("--ray_port",
                    type=str,
                    default='6380',
                    help="ray's master port.")
parser.add_argument("--vllm_tp",
                    type=int,
                    default=1,
                    help="vllm tp")
parser.add_argument("--vllm_pp",
                    type=int,
                    default=1,
                    help="vllm pp")
parser.add_argument("--vllm_mem",
                    type=float,
                    default=0.5,
                    help="vllm gpu_memory_utilization")
parser.add_argument("--ray_gpus",
                    type=int,
                    default=1,
                    help="num of each ray cluster's gpus.")
parser.add_argument("--num_vllm_engines",
                    type=int,
                    default=1,
                    help="num of vllm workers.")
parser.add_argument("--reward_func",
                    type=str,
                    default=None,
                    help="A py file include a function named `reward_func`.")
parser.add_argument("--top_p",
                    type=float,
                    default=1,
                    help="top p")
parser.add_argument("--temperature",
                    type=float,
                    default=1,
                    help="temperature")
parser.add_argument("--repetition_penalty",
                    type=float,
                    default=1,
                    help="repetition penalty")
parser.add_argument("--epsilon_low",
                    type=float,
                    default=0.2,
                    help="epsilon_low")
parser.add_argument("--epsilon_high",
                    type=float,
                    default=0.2,
                    help="epsilon_high")
parser.add_argument("--beta",
                    type=float,
                    default=0.01,
                    help="beta")   
               
parser = deepspeed.add_config_arguments(parser)
args=parser.parse_args()
assert args.early_stop != 0
assert args.max_num_checkpoints != 0
assert args.best_of>0
if args.max_num_checkpoints<0:args.best_of=1
if args.only_ckpt_lora:args.only_ckpt_model = True
args.device = deepspeed.get_accelerator().device_name()
if args.device == 'npu':
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    def get_args():
        return args

def main(args):
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    device = torch.device(args.device, args.local_rank)
    deepspeed.init_distributed(timeout=datetime.timedelta(seconds=args.timeout))
    
    args.global_rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()
    assert args.world_size % (args.pipe_parallel_size * args.tensor_parallel_size) == 0
    args.data_parallel_size = args.world_size // (args.pipe_parallel_size * args.tensor_parallel_size)
    assert args.data_parallel_size%args.sequence_parallel_size==0
    assert args.rlhf or (not args.rlhf and args.micro_batch_size==1)
    
    if args.global_batch_size is not None:
        args.gradient_accumulation_steps = max(args.global_batch_size//args.micro_batch_size//args.data_parallel_size,1)
    args.global_batch_size = args.micro_batch_size*args.data_parallel_size*args.gradient_accumulation_steps

    set_random_seed(args.seed)
    
    if args.checkpoint: 
        os.makedirs(args.checkpoint,exist_ok=True)

    if os.path.isfile(args.train_data):
        cached_dir = os.path.join(os.path.dirname(args.train_data),os.path.splitext(os.path.basename(args.train_data))[0] + f"_{os.path.basename(args.model)}")
        if args.global_rank ==0:
            from .raw2ids import write_parquet
            write_parquet(args.train_data,cached_dir,args.model,MAX_SEQ_LENGTH=args.seq_len)
        torch.distributed.barrier()
        args.train_data = cached_dir
    train_data_partitions = sorted([os.path.join(args.train_data,f) for f in os.listdir(args.train_data) if os.path.isdir(os.path.join(args.train_data,f)) and not f.startswith('.')])
    num_train_batch = 0
    seq_len = 0
    block_mask = 0
    for file in os.listdir(args.train_data):
        if file.endswith('info.json'):
            with open(os.path.join(args.train_data,file),'r') as f:
                data_info = json.load(f)
                num_train_batch += data_info['num_samples']//args.micro_batch_size//(args.data_parallel_size//args.sequence_parallel_size)
                seq_len = max(data_info['max_len'],seq_len)
                block_mask = max(data_info['max_num_blocks'],block_mask)
    num_field = len(data_info['fields'])

    args.block_mask=block_mask if args.block_mask else args.block_mask 
    args.seq_len=seq_len
    if args.force_4k:
        assert (seq_len-1)//4096>=1, 'force_4k requires seq_len>=4096.'
        args.seq_len=4097

    args.max_num_patches = 0
    args.max_num_images = 0
    if os.path.exists(f'{train_data_partitions[0]}-image.info'):
        for p in train_data_partitions:
            image_info = pyarrow.parquet.read_table(f'{p}-image.info')
            ratios = image_info['rat'].to_numpy().tolist()
            args.max_num_patches = max(int(ratios[-1][3]),args.max_num_patches)
            args.max_num_images = max(int(ratios[-1][4]),args.max_num_images)
        
    if args.eval_data:
        if os.path.isfile(args.eval_data):
            cached_dir = os.path.join(os.path.dirname(args.eval_data),os.path.splitext(os.path.basename(args.eval_data))[0] + f"_{os.path.basename(args.model)}")
            if args.global_rank ==0: 
                from .raw2ids import write_parquet
                write_parquet(args.eval_data,cached_dir,args.model,MAX_SEQ_LENGTH=args.seq_len)
            torch.distributed.barrier()
            args.eval_data = cached_dir
        eval_data_partitions = sorted([os.path.join(args.eval_data,f) for f in os.listdir(args.eval_data) if os.path.isdir(os.path.join(args.eval_data,f))])
    
    spu.initialize_sequence_parallel(args.data_parallel_size, args.pipe_parallel_size, args.tensor_parallel_size,args.sequence_parallel_size)
    
    try:
        config = AutoConfig.from_pretrained(args.model,trust_remote_code=True)
    except:
        config = AutoConfig.from_pretrained(args.model)
    config.block_mask=args.block_mask
    config.checkpoint_interval = args.checkpoint_grad_interval
    config.checkpoint_grad_step = args.no_checkpoint_grad_step
    config.num_partitions = args.emb_partitions
    config.split_dlayer = args.split_dlayer
    config.device = args.device
    config.pad_one_per_batch=args.pad_one_per_batch
    config.encoder_pipe_parallel_size = args.encoder_pipe_parallel_size
    config.lora = args.lora_dim>0
    config.lora_alpha = args.lora_alpha
    config.only_ckpt_lora = args.only_ckpt_lora
    config.one_layerspec = not args.multi_layerspec
    config.max_num_images = args.max_num_images
    config.moe_layer_pipe_size=args.moe_layer_pipe_size
    config.aux_loss_backward_scale=args.aux_loss_backward_scale
    config.swap_experts_per_steps=args.swap_experts_per_steps
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.rlhf=args.rlhf
    config.num_generations = args.num_generations
    config.max_new_tokens = args.max_new_tokens
    # config.reward_func = args.reward_func
    # config.top_p = args.top_p
    # config.repetition_penalty = args.repetition_penalty
    config.temperature = args.temperature
    
    if args.sequence_parallel_size>1: # adaptive sequence length for computation balancing
        from jllm.data.utils import get_interp_fuc
        spu.seqlens = get_interp_fuc(args.sequence_parallel_size,
                                     config.hidden_size,
                                     config.hidden_size*config.num_key_value_heads//config.num_attention_heads,
                                     args.seq_len-1,
                                     config.intermediate_size,
                                     attention_alpha = args.attention_alpha-int(args.tensor_parallel_size>1),
                                     dynamic=num_field>1)
        config.seq_len = spu.seqlens[spu.get_sequence_parallel_rank()+1](args.seq_len-1)-spu.seqlens[spu.get_sequence_parallel_rank()](args.seq_len-1)
    else:
        spu.seqlens = None
        config.seq_len = args.seq_len-1 # (args.seq_len-1+args.sequence_parallel_size-1)//args.sequence_parallel_size+1
    args.seq_len = config.seq_len
    if hasattr(config,'num_experts_per_tok'):
        config.split_dlayer = True
        if not hasattr(config,'partition_method') or len(config.partition_method)!=args.pipe_parallel_size+1:
            config.partition_method =autopartition_transformer(config,args,get_virtual_num_hidden_layers[config.architectures[0]](config))
    elif not hasattr(config,'partition_method') or len(config.partition_method)!=args.pipe_parallel_size+1:
        if hasattr(config,'llm_config'):
            config.partition_method = autopartition_decoder(config.llm_config,args)
        elif hasattr(config,'vision_config') and not hasattr(config,'llm_config'):
            config.partition_method = autopartition_decoder(config,args)
        else:
            config.partition_method = autopartition_transformer(config,args)
    
    if args.from_ckpt or args.resume_ckpt:
        temp = args.from_ckpt or args.resume_ckpt
        if args.tag is None:
            with open(os.path.join(temp,'latest'),'r') as f:
                args.tag = f.read().strip()
        temp = os.path.join(temp,args.tag)
        config.partition_method = AutoConfig.from_pretrained(temp,trust_remote_code=True).partition_method
    
    if isinstance(config.partition_method,str) and ',' not in config.partition_method:
        partition_method = config.partition_method
    elif config.one_layerspec :
        if isinstance(config.partition_method,str):
            config.partition_method = config.partition_method.split(',')
        if args.pipe_parallel_size == 1:
            partition_method = 'uniform'
        elif hasattr(config,'vision_config') and args.encoder_pipe_parallel_size == 0:
            partition_method = str([0]+list(range(2,2+len(config.partition_method))))[1:-1]
        else:
            partition_method = str(list(range(args.encoder_pipe_parallel_size+len(config.partition_method))))[1:-1]
    else:
        partition_method = str(config.partition_method)[1:-1]
        
    if args.expert_parallel_size==1:
        topo = ProcessTopology(['data','pipe','model'], [args.data_parallel_size, args.pipe_parallel_size, args.tensor_parallel_size])
    else:
        topo = ProcessTopology(['pipe','data','model'], [args.pipe_parallel_size, args.data_parallel_size, args.tensor_parallel_size])
    args.seed = args.seed + topo.get_coord(args.global_rank).pipe
    
    if args.rlhf:
        assert args.num_generations//args.sequence_parallel_size%args.micro_batch_size == 0
        assert (args.gradient_accumulation_steps*args.micro_batch_size)%(args.num_generations//args.sequence_parallel_size)==0
        num_train_batch = num_train_batch*args.num_generations
        config.micro_batch_size = args.micro_batch_size
        config.epsilon_low = args.epsilon_low
        config.epsilon_high = args.epsilon_high
        config.beta = args.beta
        from jllm import rlhf
        from vllm.utils import get_ip
        args.ray_ip = get_ip() if args.ray_ip is None else args.ray_ip
        dp_rank = topo.get_coord(rank=args.global_rank).data
        args.vllm_engine_rank = dp_rank%args.num_vllm_engines
        if args.isolated_vllm:
            rlhf.connect_vllm_actor(f"{args.ray_ip}:{args.ray_port}",args.vllm_engine_rank)
        else:
            from jllm.vllm import init_vllm
            if args.global_rank==0:
                vllm_actor = init_vllm(f"{args.ray_ip}:{args.ray_port}",
                                       args.model,
                                       gpu_memory_utilization=args.vllm_mem,
                                       tensor_parallel_size=args.vllm_tp,
                                       pipeline_parallel_size=args.vllm_pp,
                                       gpus=args.ray_gpus,
                                       max_num_batched_tokens=args.num_generations*args.max_new_tokens+args.seq_len,
                                       max_model_len =args.max_new_tokens+args.seq_len)
                rlhf.vllm_actor=vllm_actor
            torch.distributed.barrier()
            if args.global_rank!=0:
                args.num_vllm_engines = args.ray_gpus//vllm_tp//vllm_tp
                rlhf.connect_vllm_actor(f"{args.ray_ip}:{args.ray_port}",args.vllm_engine_rank)

    train_ds_config = get_train_ds_config
    if args.ds_config is not None:train_ds_config = dynamic_import_module(args.ds_config).get_train_ds_config
    ds_config = train_ds_config(offload=False,stage=args.zero_stage,)
    
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.micro_batch_size
    ds_config[
        'train_batch_size'] = args.global_batch_size//args.sequence_parallel_size
    ds_config['steps_per_print'] = args.steps_per_print

    if args.tensor_parallel_size>1 or args.expert_parallel_size>1 or args.moe_layer_pipe_size>2:
        if args.device == 'npu':
            import jllm.ascend
        from jllm.core import parallel_state,tensor_parallel
        if args.expert_parallel_size==1:
            parallel_state.initialize_model_parallel(args.tensor_parallel_size,
                                                     args.pipe_parallel_size,
                                                     order='tp-pp-dp')
        else:
            parallel_state.initialize_model_parallel(args.tensor_parallel_size,
                                                     args.pipe_parallel_size,
                                                     expert_model_parallel_size=args.expert_parallel_size)
        tensor_parallel.model_parallel_cuda_manual_seed(args.seed)
        from jllm.core.model_parallel_config import ModelParallelConfig
        parallel_config = ModelParallelConfig(tensor_model_parallel_size=args.tensor_parallel_size,
                                              pipeline_model_parallel_size=args.pipe_parallel_size,
                                              params_dtype=config.torch_dtype,
                                              pipeline_dtype=config.torch_dtype,
                                              async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce
                                             )
        parallel_state.set_aux_loss_alpha(args.aux_loss_alpha)
        parallel_config.seq_length = config.seq_len
        parallel_config.max_num_patches = args.max_num_patches
        parallel_config.padding_rate = args.padding_rate
        # parallel_config.sequence_parallel = args.sequence_parallel
        from jllm.model import ModelParallel
        with deepspeed.zero.Init(data_parallel_group=parallel_state.get_data_parallel_group(),
                         config_dict_or_path=ds_config,
                         enabled=args.zero_stage == 3,
                         mpu=parallel_state):
            model =  ModelParallel[config.architectures[0]](
                config,
                parallel_config=parallel_config,
                topology=topo,
                base_seed=args.seed,
                partition_method=partition_method,
                )
    else:
        model = ModelPipe[config.architectures[0]](
            config,
            topology=topo,
            base_seed=args.seed,
            partition_method=partition_method,
            )
    
    if not(args.resume_ckpt or args.from_ckpt) and not args.init: 
        model.from_pretrained(args.model,args.cache_model)
        if args.only_cache_model:
            return
    
    if args.lora_dim > 0:
        if args.tensor_parallel_size > 1:
            from peft import LoraConfig, inject_adapter_in_model
            lora_config = LoraConfig(
                r=args.lora_dim,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_module_name.split(','),
                megatron_config=parallel_config,
                megatron_core="jllm.core",
                )
            model = inject_adapter_in_model(lora_config,model)
        else:
            model = convert_linear_layer_to_lora(
                model,
                args.lora_module_name.split(','),
                args.lora_dim,
                args.lora_alpha)
            if args.only_optimize_lora:
                model = only_optimize_lora_parameters(model)
                model = make_model_gradient_checkpointing_compatible(model)
                
    if "optimizer" not in ds_config:
        optimizer_grouped_parameters = get_param_groups(model)
        if args.expert_parallel_size>1:
            from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
            optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(optimizer_grouped_parameters)
        optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95))
                              
    '''
    How many folders, how many partitions. 
    If you want to load the data into memory at one time, moving all the parquet files to same folder. 
    That may cause "num_update_steps_per_epoch" to be un-precision. But it donesn't matter.
    ''' 
    num_train_batch *=((seq_len-1)//4096 if args.force_4k else 1)
    num_update_steps_per_epoch = num_train_batch // args.gradient_accumulation_steps + len(train_data_partitions) - 1
    args.num_training_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
    args.aux_loss_step = int(args.num_training_steps*args.aux_loss_step if args.aux_loss_step<=1 else args.aux_loss_step)
    if 'scheduler' not in ds_config:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps if args.num_warmup_steps >= 1 else int(args.num_warmup_steps * num_update_steps_per_epoch),
            num_training_steps=args.num_training_steps)
    
    engine, *_ = deepspeed.initialize(
        args = args,
        config = ds_config,
        model = model if not args.rlhf else rlhf.make_duplicate_module(model),
        optimizer = optimizer if "optimizer" not in ds_config else None,
        lr_scheduler = lr_scheduler if "scheduler" not in ds_config else None,
        )
    engine.set_train_batch_size(engine.train_batch_size()*args.sequence_parallel_size)
    
    train(args,engine,train_data_partitions,eval_data_partitions if args.eval_data else None)

if __name__ == "__main__":
    import jllm.deepspeed_patch
    main(args)
