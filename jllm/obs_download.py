import os
import numpy as np
import argparse
import moxing as mox
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import time
# import torch
# import torch_npu
# from datetime import timedelta
# from torch_npu.contrib import transfer_to_npu

def obs_download(rank,world_size,pp,tp,model,data):
    
    topo=np.arange(world_size).reshape(-1,pp,tp)
    dr,pr,tr=np.where(topo==rank)
    mr=pr*tp+tr
    
    opt= 'bf16_zero_pp_rank_{dr}_mp_rank_{mr:02d}_optim_states.pt'
    md = 'layer_{pr:02d}-model_{tr:02d}-model_states.pt'
    ms = 'mp_rank_{mr:02d}_model_states.pt'
    
    downloads = [
        opt.format(dr=dr.item(),mr=mr.item()),
        md.format(pr=pr.item(),tr=tr.item()),
        ms.format(mr=mr.item())
    ]
    
    if tp>1:
        tp_st = "tensor-{tr:02d}-of-{tp:02d}-pipeline-{pr:02d}-of-{pp:02d}.safetensors"
        downloads.append(tp_st.format(tr=tr.item()+1,tp=tp,pr=pr.item()+1,pp=pp))
    elif pp>1:
        pp_st = "model-{pr:05d}-of-{pp:05d}.safetensors"
        downloads.append(pp_st.format(pr=pr.item()+1,pp=pp))
    else:
        downloads.append('model.safetensors')
    downloads.append('config.json')
    
    downloaded = []
    for file in downloads:
        file_path = os.path.join(model,file)
        if mox.file.exists(file_path):
            if file.endswith('safetensors') or (file=='config.json' and rank%8==0):
                mox.file.copy(file_path,os.path.join('/cache/model',file))
                downloaded.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+": "+file)
            elif file.endswith('pt'):
                mox.file.copy(file_path,os.path.join('/cache/model/1',file))
                downloaded.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+": 1/"+file)
    if rank%8==0 and os.path.exists('/cache/model/1'):
        mox.file.copy(file_path,os.path.join('/cache/model/1',file))
        downloaded.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+": 1/"+file)
        with open('/cache/model/latest','w') as f:f.write('1')
    if data is not None and mox.file.exists(data):
        if pr==0 and tr ==0:
            mox.file.copy_parallel(data,'/cache/data')
            downloaded.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+": "+data)
        elif rank%8==0:
            for file in mox.file.list_directory(data, recursive=False):
                if file[-4:]=='.crc':
                    mox.file.copy(os.path.join(data,file),os.path.join('/cache/data',file))
                    downloaded.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+": "+file)
                else:
                    os.makedirs(os.path.join('/cache/data',file),exist_ok=True)
    return downloaded

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pp', type=int,default=1,help='pp size' )
    parser.add_argument('--tp', type=int,default=1,help='tp size' )
    parser.add_argument('--data', type=str,help='train data obs path')
    parser.add_argument('--model', type=str,help='model obs path')
    parser.add_argument('--sleep', type=int,default=30,help='sleep seconds')
    args = parser.parse_args()
    
    # init_process_group_kwargs = {
        # 'rank' : int(os.environ.get("RANK", 0)),
        # 'world_size' : int(os.environ.get("WORLD_SIZE", 1)),
        # 'backend' : 'hccl',
        # 'timeout': timedelta(minutes=args.timeout),
        # }

    # torch.distributed.init_process_group(**init_process_group_kwargs)
    # world_size = torch.distributed.get_world_size()
    # rank = torch.distributed.get_rank()
    
    #obs_download(rank,world_size,args.pp,args.tp,args.model,args.data)
    print("下载开始时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    with ProcessPoolExecutor(max_workers=8) as exe:
        func = partial(obs_download,
                       world_size=int(os.environ["WORLD_SIZE"]),
                       pp=args.pp,
                       tp=args.tp,
                       model=args.model,
                       data=args.data)
        NODE_RANK = int(os.environ["NODE_RANK"])*8
        result = list(exe.map(func,range(NODE_RANK,NODE_RANK+8)))
        
    sync_file='/'.join(args.model.rsplit(os.path.sep)[:3]+['sync',f'{int(os.environ["NODE_RANK"]):04}.txt'])
    with mox.file.File(sync_file, 'w') as f:
        for records in result:
            f.write('\n'.join(records)+'\n')
    print("下载完成时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    while len(mox.file.list_directory(os.path.dirname(sync_file), recursive=False)) != int(os.environ["WORLD_SIZE"])//8:
        time.sleep(args.sleep)
    #torch.distributed.barrier()
    