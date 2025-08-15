import argparse
import os
from jllm.obs_download import obs_exists,obs_copy

if __name__=='__main__'	:
    parser = argparse.ArgumentParser()
    parser.add_argument('-O','--obs_path', type=str,default=None,help='obs path')
    parser.add_argument('-H','--hash_file', type=str,default=None,help='hash file')
    parser.add_argument('-N','--node_num', type=int,default=1,help='number of infer nodes')
    args = parser.parse_args()
    
    node_rank = int(os.environ["NODE_RANK"])
    nnodes = int(os.environ["NNODES"])
    infer_start = nnodes - args.node_num
    
    ip_path = os.path.join(args.obs_path ,'rank_ip',args.hash_file,'ip.txt')
    local_file='/tmp/ip.txt'
    if node_rank == infer_start:
        from vllm.utils import get_ip
        ip = get_ip()
        with open(local_file, 'w') as f:
            f.write(ip)
        obs_copy(local_file, ip_path)
    else:
        import time
        while not obs_exists(ip_path):
            time.sleep(5)
        obs_copy(ip_path,local_file)