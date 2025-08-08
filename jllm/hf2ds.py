import json
import argparse
from transformers import AutoConfig
from jllm.model import (autopartition_transformer,
                        get_virtual_num_hidden_layers,
                        get_layer_map,
                        ModelPipe,
                        ModelParallel)
from safetensors.torch import save_file, load_file
from functools import partial
save_file=partial(save_file,metadata={'format': 'pt'})
from concurrent.futures import ProcessPoolExecutor
import gc,os,tqdm
import torch

'''
 python -m jllm.hf2ds -p 16 -t 8 -e 4 --partition_method 8,6 -m unsloth/DeepSeek-R1 -o cached_model
'''

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pipe_parallel_size', type=int,default=1,help='pp size' )
    parser.add_argument('-t','--tensor_parallel_size', type=int,default=1,help='tp size')
    parser.add_argument('-e','--expert_parallel_size', type=int,default=1,help='ep size')
    parser.add_argument('--moe_layer_pipe_size', type=int,default=2,help='moe size')
    parser.add_argument('--partition_method', type=str,default='fast',help='partition method')
    parser.add_argument('-m','--model', type=str,help='model obs path')
    parser.add_argument('-o','--output', type=str,help='output')
    args = parser.parse_args()
    os.makedirs(args.output,exist_ok=True)
    args.seq_len=4096
    pipe_size=args.pipe_parallel_size
    tensor_size=args.tensor_parallel_size
    expert_size=args.expert_parallel_size
    tensor_expert_size = args.tensor_parallel_size*args.expert_parallel_size
    try:
        config = AutoConfig.from_pretrained(args.model,trust_remote_code=True)
    except:
        config = AutoConfig.from_pretrained(args.model)
    config.moe_layer_pipe_size=args.moe_layer_pipe_size
    moe = hasattr(config,'num_experts_per_tok')
    if os.path.exists(os.path.join(args.model,"model.safetensors.index.json")):
        with open(os.path.join(args.model,"model.safetensors.index.json"),"r") as f: 
            weight_map = json.load(f)["weight_map"]
    else:
        class UniversalContainer:
            def __contains__(self, key):
                return True
            def __getitem__(self, key):
                return 'model.safetensors'
        weight_map = UniversalContainer()
    if moe:
        layer_map = get_layer_map[config.architectures[0]](config)
        num_expert_per_group = (config.num_experts if hasattr(config,'num_experts') else config.n_routed_experts)//(config.moe_layer_pipe_size-1)
        num_expert_per_rank = num_expert_per_group//tensor_expert_size
        partitions = autopartition_transformer(config,args,get_virtual_num_hidden_layers[config.architectures[0]](config))
    else:
        partitions = autopartition_transformer(config,args)
    if tensor_size>1 or expert_size>1 or args.moe_layer_pipe_size>2:
        model = ModelParallel[config.architectures[0]]
    else:
        model = ModelPipe[config.architectures[0]]
        
    print(f'partitions:{partitions}')
    def hf2ds(p):
        for te in range(tensor_expert_size):
            if moe:
                pipe2hf=model.get_pipe2hf(p,
                                          te,
                                          partitions,
                                          layer_map,
                                          num_expert_per_rank,
                                          num_expert_per_group,
                                          getattr(config,'num_nextn_predict_layers',0))
            else:
                pipe2hf=model.get_pipe2hf(p,
                                          te,
                                          config.num_hidden_layers,
                                          config.attention_bias,
                                          partitions)
            e=te//tensor_size
            state_dict={}
            source_dict={}
            local_hks = set()
            for hf_k in pipe2hf.values():
                if e==0:
                    if isinstance(hf_k,tuple):
                        local_hks.update(hf_k)
                    else:
                        local_hks.add(hf_k)
                else:
                    if isinstance(hf_k,tuple) and 'mlp.experts' in hf_k[0]:
                        local_hks.update(hf_k)
            for part in tqdm.tqdm({weight_map[k] for k in local_hks if k in weight_map}):
                tmp = load_file(os.path.join(args.model,part))
                for k in local_hks & tmp.keys():
                    source_dict[k] = tmp.pop(k)
                del tmp
                gc.collect()
            t=te%tensor_size
            model.get_weights_(pipe2hf,state_dict,source_dict,t,tensor_size)
            if moe:
                model_file=f"tensor-{t+1:02d}-of-{tensor_size:02d}-expert-{e+1:02d}-of-{expert_size:02d}-pipeline-{p+1:02d}-of-{pipe_size:02d}.safetensors"
            elif tensor_size>1:
                model_file=f"tensor-{t+1:02d}-of-{tensor_size:02d}-pipeline-{p+1:02d}-of-{pipe_size:02d}.safetensors"
            elif pipe_size>1:
                model_file=f"model-{p+1:05d}-of-{pipe_size:05d}.safetensors"
            else:
                model_file="model.safetensors"
            if state_dict:save_file(state_dict,os.path.join(args.output,model_file))
            print(f'saved {model_file}')
        
    with ProcessPoolExecutor(max_workers=args.pipe_parallel_size) as exe:
        list(exe.map(hf2ds,range(args.pipe_parallel_size)))
    config.partition_method = partitions
    config.to_json_file(os.path.join(args.output,"config.json"))
