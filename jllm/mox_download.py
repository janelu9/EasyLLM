import os
import numpy as np
import argparse
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import time
import moxing as mox

def obs_download(rank,world_size,args):
    pp,ep,tp,model,only_model,data=args.pp,args.ep,args.tp,args.model,args.only_model,args.data
    if ep ==1:
        topo=np.arange(world_size).reshape(-1,pp,tp)
        dr,pr,tr=np.where(topo==rank)
        er = 0
    else:
        topo=np.arange(world_size).reshape(pp,-1,tp)
        pr,dr,tr = np.where(topo==rank)
        _,_,er,_ = np.where(topo.reshape(pp,-1,ep,tp)==rank)
        er=er.item()
    dr=dr.item()
    pr=pr.item()
    tr=tr.item()
    downloaded = []
    if model is not None:
        
        mr=pr*tp+tr
        
        opt = 'bf16_zero_pp_rank_{dr}_mp_rank_{mr:02d}_optim_states.pt'
        md = 'layer_{pr:02d}-model_{tr:02d}-model_states.pt'
        ms = 'mp_rank_{mr:02d}_model_states.pt'
        
        ems = '_expert_{er}_mp_rank_{mr:02d}_model_states.pt'
        eopt = 'expp_rank_{dr}_mp_rank_{mr:02d}_optim_states.pt'
        
        downloads = [
            md.format(pr=pr,tr=tr),
            ms.format(mr=mr),
        ]
        if not only_model:
            downloads.append(opt.format(dr=dr,mr=mr))
            if ep>1:downloads.append(eopt.format(dr=dr,mr=mr))
        if ep>1:
            ep_st= "tensor-{tr:02d}-of-{tp:02d}-expert-{er:02d}-of-{ep:02d}-pipeline-{pr:02d}-of-{pp:02d}.safetensors"
            downloads.append(ep_st.format(tr=tr+1,tp=tp,er=er+1,ep=ep,pr=pr+1,pp=pp))
            if er!=0:downloads.append(ep_st.format(tr=tr+1,tp=tp,er=1,ep=ep,pr=pr+1,pp=pp))
            downloads.extend([f for f in mox.file.list_directory(model, recursive=False) if ems.format(er=er,mr=mr) in f])
        elif tp>1:
            tp_st = "tensor-{tr:02d}-of-{tp:02d}-pipeline-{pr:02d}-of-{pp:02d}.safetensors"
            downloads.append(tp_st.format(tr=tr+1,tp=tp,pr=pr+1,pp=pp))
        elif pp>1:
            pp_st = "model-{pr:05d}-of-{pp:05d}.safetensors"
            downloads.append(pp_st.format(pr=pr+1,pp=pp))
        else:
            downloads.append('model.safetensors')
        downloads.append('config.json')
        
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
                if file[-5:] in {'.json','.info'}:
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
    parser.add_argument('--only_model', action='store_true',help='only download model')
    parser.add_argument('--sleep', type=int,default=30,help='sleep seconds')
    args = parser.parse_args()
    
    NODE_RANK = int(os.environ["NODE_RANK"])
    sync_dir = '/'.join((args.model if args.model is not None else args.data).rsplit(os.path.sep)[:3]+['sync'])
    if NODE_RANK==0 and mox.file.exists(sync_dir):
        mox.file.remove(sync_dir, recursive=True)

    print("下载开始时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    with ProcessPoolExecutor(max_workers=8) as exe:
        func = partial(obs_download,
                       world_size=int(os.environ["WORLD_SIZE"]),
                       args=args)
        result = list(exe.map(func,range(NODE_RANK*8,NODE_RANK*8+8)))
    sync_file=os.path.join(sync_dir,f'{NODE_RANK:04}.txt')
    with mox.file.File(sync_file, 'w') as f:
        for records in result:
            f.write('\n'.join(records)+'\n')
    print("下载完成时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    while len(mox.file.list_directory(sync_dir, recursive=False)) != int(os.environ["WORLD_SIZE"])//8:
        time.sleep(args.sleep)
    