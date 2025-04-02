import json
import argparse
from transformers import AutoConfig
from jllm.model.deepseek_v3.parallel_deepseek_v3 import get_layer_map,DeepseekV3ForCausalLM
from jllm.model import autopartition_transformer,deepseek_v3_get_num_hidden_layers
from safetensors.torch import save_file, load_file
from functools import partial
save_file=partial(save_file,metadata={'format': 'pt'})
from concurrent.futures import ProcessPoolExecutor
import gc,os,tqdm
import torch

'''
python -m jllm.hf2ds -p 41 -t 8 -m unsloth/DeepSeek-R1 -o cached_model
'''

def get_weights_(pipe2hf,state_dict,tmp,tensor_rank,tensor_size):
    for pk,hk in pipe2hf.items():
        if hk in tmp:
            tensor = tmp.pop(hk)
            if "embed_tokens" in hk or "lm_head" in hk or 'q_b_proj' in hk or 'kv_b_proj' in hk:
                tensor = tensor.chunk(tensor_size,0)[tensor_rank].contiguous()
            elif "o_proj" in hk or "mlp.down_proj" in hk:
                tensor = tensor.chunk(tensor_size,1)[tensor_rank].contiguous()
            state_dict[pk] = tensor
        elif "gate_up_proj" in pk and hk[0] in tmp:
            state_dict[pk] = torch.cat([tmp.pop(k).chunk(tensor_size,0)[tensor_rank] for k in hk],0).contiguous()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pipe_parallel_size', type=int,default=1,help='pp size' )
    parser.add_argument('-t','--tensor_parallel_size', type=int,default=1,help='tp size')
    parser.add_argument('--moe_layer_pipe_size', type=int,default=5,help='moe size')
    parser.add_argument('--partition_method', type=str,default='13,7',help='partition method')
    parser.add_argument('-m','--model', type=str,help='model obs path')
    parser.add_argument('-o','--output', type=str,help='output')
    args = parser.parse_args()

    args.seq_len=4096
    pipe_size=args.pipe_parallel_size
    tensor_size=args.tensor_parallel_size
    try:
        config = AutoConfig.from_pretrained(args.model,trust_remote_code=True)
    except:
        config = AutoConfig.from_pretrained(args.model)
    config.moe_layer_pipe_size=args.moe_layer_pipe_size
    with open(os.path.join(args.model,"model.safetensors.index.json"),"r") as f: 
        weight_map = json.load(f)["weight_map"]
    layer_map = get_layer_map(config)
    num_expert_per_group = config.n_routed_experts//(config.moe_layer_pipe_size-1)
    num_expert_per_rank = num_expert_per_group//args.tensor_parallel_size
    partitions = autopartition_transformer(config,args,deepseek_v3_get_num_hidden_layers(config))
    print(f'partitions:{partitions}')
    def hf2ds(p):
        for t in range(args.tensor_parallel_size):
            pipe2hf=DeepseekV3ForCausalLM.get_pipe2hf(
                                                       p,
                                                       t,
                                                       partitions,
                                                       layer_map,
                                                       num_expert_per_rank,
                                                       num_expert_per_group)
                                     
            state_dict={}
            source_dict={}
            local_hks = set()
            for hf_k in pipe2hf.values():
                if isinstance(hf_k,tuple):
                    local_hks.update(hf_k)
                else:
                    local_hks.add(hf_k)
            # with open(os.path.join(args.output,f"tensor-{t + 1:02d}-of-{tensor_size:02d}-pipeline-{p + 1:02d}-of-{pipe_size:02d}.safetensors" ),'w')as f:
                # f.write('\n'.join(local_hks)+'\n')
            for part in tqdm.tqdm({weight_map[k] for k in local_hks}):
                tmp = load_file(os.path.join(args.model,part))
                for k in local_hks & tmp.keys():
                    source_dict[k] = tmp.pop(k)
                del tmp
                gc.collect()
            get_weights_(pipe2hf,state_dict,source_dict,t,tensor_size)
            model_file=os.path.join(args.output,f"tensor-{t + 1:02d}-of-{tensor_size:02d}-pipeline-{p + 1:02d}-of-{pipe_size:02d}.safetensors" )
            save_file(state_dict,model_file)
            print(f'saved {model_file}')
        
    with ProcessPoolExecutor(max_workers=args.pipe_parallel_size) as exe:
        list(exe.map(hf2ds,range(args.pipe_parallel_size)))
