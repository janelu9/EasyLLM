def parallel_type(k):
    if  "o_proj.weight" in k or ("down_proj.weight" in k and 'shared_experts' not in k) or 'attn.proj.weight' in k or 'mlp.fc2.weight' in k or 'mlp.2.weight' in k :
        return 1
    if "lm_head" in k or ("gate_proj" in k and 'shared_experts' not in k) or ("up_proj" in k and 'shared_experts' not in k) or "embed_tokens" in k or "q_proj" in k \
    or "k_proj" in k or "v_proj" in k or 'attn.qkv' in k or 'mlp.fc1' in k or 'mlp.0' in k \
    or 'q_b_proj' in k or 'kv_b_proj' in k:
        return 0
    return -1


if __name__=='__main__':
    import os
    import gc
    from concurrent.futures import ProcessPoolExecutor
    import torch
    import tqdm
    import json
    import argparse
    from safetensors.torch import save_file, load_file
    from functools import partial
    save_file=partial(save_file,metadata={'format': 'pt'})
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-C','--ckpt', type=str, help="checkpoint.")
    parser.add_argument('-H','--hf',type=str,default="",help="Where to store the model.")
    args = parser.parse_args()
    args.hf = (args.ckpt+"_hf") if not args.hf else args.hf
    os.makedirs(args.hf,exist_ok=True)
    ckpt_path = args.ckpt
    files = os.listdir(ckpt_path)
    
    files = [f for f in files if f.endswith('safetensors')]
    files.sort()
    num_stages = int(files[0][:-12].rsplit('-',1)[1])
    
    def func(pipe_rank):
        
        pts = [load_file(os.path.join(ckpt_path,f)) for f in files if int(f.rsplit('-',3)[1])==pipe_rank]
        moe = {k:p.pop(k) for p in pts for k in set(p.keys()) if '.experts.' in k}
        pts = [p for p in pts if p]
        
        keys = set(pts[0].keys())
        state_dict ={}
        
        index = {"metadata":{"total_size":0},"weight_map":{}}
        model_file =f"model-{pipe_rank:05d}-of-"+f"{num_stages:05d}.safetensors"
        
        for k in tqdm.tqdm(keys):
            dim = parallel_type(k)
            if dim >=0:
                state_dict[k] = torch.cat([p.pop(k) for p in pts],dim)
            else:
                state_dict[k] = pts[0].pop(k)
            index["metadata"]["total_size"] +=state_dict[k].nbytes
            index["weight_map"].update({k:model_file})
        for k in set(moe.keys()):
            state_dict[k] = moe.pop(k)
            index["metadata"]["total_size"] +=state_dict[k].nbytes
            index["weight_map"].update({k:model_file})
        del pts
        gc.collect()
        save_file(state_dict,os.path.join(args.hf,model_file if num_stages>1 else "model.safetensors"))
        print(f'{model_file} saved.') if num_stages>1 else print(f'model.safetensors saved.')
        return index

    with ProcessPoolExecutor(max_workers=min(num_stages,32)) as exe:
        res = list(exe.map(func,list(range(1,num_stages+1))))
        
    if num_stages>1:
        index = {"metadata":{"total_size":0},"weight_map":{}}
        for r in res:
            index["metadata"]["total_size"] += r["metadata"]["total_size"]
            index["weight_map"].update(r["weight_map"])
        with open(os.path.join(args.hf ,"model.safetensors.index.json"),'w') as f:
            json.dump(index,f,indent=2)
    os.system(f'cp -f {args.ckpt}/config.json {args.hf}/')
    print("Done!")