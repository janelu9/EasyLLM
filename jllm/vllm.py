import os
import torch
import ray
from vllm import LLM
from jllm.cat2hf import parallel_type
from jllm.rlhf import stateless_init_process_group

if hasattr(torch,'npu'):
    NPU=True
    #import subprocess
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
else:
    NPU=False

# def get_npu_uuid(index):
    # cmd = ['npu-smi', 'info','-t','board','-i',str(index)]
    # result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # return result.stdout.strip().split('\n\t')[4].split(' ')[-1]

class WorkerExtension:

    def report_device_id(self) -> str:
        from vllm.platforms import current_platform
        from vllm.distributed import parallel_state as mpu
        self.device_uuid = '1' if NPU else current_platform.get_device_uuid(self.device.index) 
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.pp_size = mpu.get_pp_group().world_size
        self.pp_rank = mpu.get_pp_group().rank_in_group
        self.model_update_groups={}
        return self.device_uuid
        
    def report_pipeline_size(self):
        return self.pp_size
    
    def report_pp_rank_of_global_rank_in_vllm(self, device_uuid):
        if device_uuid == self.device_uuid:
            return self.pp_rank
        return -1
    
    def update_weight_by_cpu_handles(self, cpu_handles,tile=True):
        weights =[]
        for name, handle in cpu_handles:
            func, args = handle
            tensor = func(*args)
            if self.tp_size>1 and tile:
                dim = parallel_type(name)
                if dim==0:
                    tensor=tensor.tile(self.tp_size,1) if tensor.dim()==2 else tensor.tile(self.tp_size)
                elif dim==1:
                    tensor=tensor.tile(1,self.tp_size) if tensor.dim()==2 else tensor.tile(self.tp_size)
            weights.append((name, tensor))
        self.model_runner.model.load_weights(weights=weights)

    def init_weight_update_group(self, 
                                 master_address,
                                 master_port,
                                 global_rank,
                                 world_size,
                                 rank_offset=1,
                                 pp_rank_of_global_rank_in_vllm=-1,
                                 train_tp_rank=0):
        from vllm.distributed.parallel_state import get_world_group
        rank = self.pp_rank
        if rank != pp_rank_of_global_rank_in_vllm and train_tp_rank==self.tp_rank:
            if rank < pp_rank_of_global_rank_in_vllm or pp_rank_of_global_rank_in_vllm == -1:
                rank += rank_offset
            self.model_update_groups[global_rank] = stateless_init_process_group(
                master_address,
                master_port,
                rank,
                world_size,
                torch.cuda.current_device(),
            )

    def update_weight_by_nccl(self,
                              global_rank,
                              name, dtype, shape, 
                              pp_rank_of_global_rank_in_vllm=-1,
                              train_tp_rank=0,
                              tile=True):
        from vllm.distributed.parallel_state import get_world_group
        if self.pp_rank != pp_rank_of_global_rank_in_vllm and train_tp_rank==self.tp_rank:
            weight = torch.empty(shape, dtype=dtype, device="cuda")
            self.model_update_groups[global_rank].broadcast(weight,src=0,stream=torch.cuda.current_stream())
            if self.tp_size>1 and tile:
                dim = parallel_type(name)
                if dim==0:
                    weight = weight.tile(self.tp_size,1) if weight.dim()==2 else weight.tile(self.tp_size)
                elif dim==1:
                    weight = weight.tile(1,self.tp_size) if weight.dim()==2 else weight.tile(self.tp_size)
            self.model_runner.model.load_weights(weights=[(name, weight)])
            del weight

    def update_weights_by_ipc_handles(self, ipc_handles,tile=True):
        if self.device_uuid in ipc_handles:
            handles = ipc_handles[self.device_uuid]
            device_id = self.device.index
            weights = []
            for name, handle in handles:
                func, args = handle
                list_args = list(args)
                list_args[6] = device_id
                tensor = func(*list_args)
                if self.tp_size>1 and tile:
                    dim = parallel_type(name)
                    if dim==0:
                        tensor = tensor.tile(self.tp_size,1) if tensor.dim()==2 else tensor.tile(self.tp_size)
                    elif dim==1:
                        tensor = tensor.tile(1,self.tp_size) if tensor.dim()==2 else tensor.tile(self.tp_size)
                weights.append((name, tensor))
            self.model_runner.model.load_weights(weights=weights)
            torch.cuda.synchronize()

class vLLM(LLM):
    def __init__(self, *args,bundle_indices, **kwargs):
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        if not NPU:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.6"
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        super().__init__(*args, **kwargs)
        
def init_vllm(address,
              model,
              gpu_memory_utilization=0.5,
              tensor_parallel_size=1,
              pipeline_parallel_size=1,
              gpus=1,
              max_num_batched_tokens=1024,
              max_model_len=1024):
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    
    ray.init(address=address,ignore_reinit_error=True)
    if NPU:
        distributed_executor_backend = 'uni' if tensor_parallel_size==1 else 'mp'
    else:
        pg = placement_group([{"GPU": 1, "CPU": 0}]*gpus)
        ray.get(pg.ready())
        distributed_executor_backend = 'ray'
    
    model_size = tensor_parallel_size*pipeline_parallel_size
    num_engines = gpus//model_size
    actor_pool = []
    for i in range(num_engines):
        vllm_actor = ray.remote(
            num_cpus=0,
            num_gpus=0,
            scheduling_strategy=None if NPU else PlacementGroupSchedulingStrategy(placement_group=pg,placement_group_capture_child_tasks=True),
            resources= {"NPU" : model_size} if NPU else None
        )(vLLM).options(name=f"llm_{i}",namespace="vllm", lifetime="detached").remote(
            model=model,
            enforce_eager=True,
            worker_extension_cls="jllm.vllm.WorkerExtension",
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            bundle_indices=None if NPU else list(range(i*model_size,(i+1)*model_size)),
        )
        ray.get(vllm_actor.collective_rpc.remote("report_device_id"))
        actor_pool.append(vllm_actor)
    return actor_pool
    
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from vllm.inputs import TokensPrompt
from vllm import SamplingParams

app = FastAPI()

class GenerationRequest(BaseModel):
    rank: int = 0
    prompts: List[List[int]] = [[]]
    sampling_params: Dict[str, Any] = {}
    
@app.post("/generate")
async def generate(request: GenerationRequest):
    token_prompts = [TokensPrompt(prompt_token_ids=p) for p in request.prompts]
    actor = actor_pool[request.rank]
    outputs = ray.get(actor.generate.remote(
                      prompts=token_prompts,
                      sampling_params=SamplingParams(**request.sampling_params)))
    text = [oi.text for o in outputs for oi in o.outputs]
    token_ids = [oi.token_ids for o in outputs for oi in o.outputs]
    return JSONResponse({'text':text,'token_ids':token_ids})
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='vllm')
    parser.add_argument("--model",
                        type=str,
                        default= "",
                        help="huggingface's model path")
    parser.add_argument("--max_prefill_len",
                        type=int,
                        default=128,
                        help="max prefill length")
    parser.add_argument("--num_generations",
                        type=int,
                        default=4,
                        help="num generations")
    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=33,
                        help="max new tokens")
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
                        
    args=parser.parse_args()
    args.model=os.path.abspath(args.model)
    from vllm.utils import get_ip
    ray_ip = get_ip() if args.ray_ip is None else args.ray_ip
    # try:
    actor_pool=init_vllm(f"{ray_ip}:{args.ray_port}",
                        args.model,
                        gpu_memory_utilization=args.vllm_mem,
                        tensor_parallel_size=args.vllm_tp,
                        pipeline_parallel_size=args.vllm_pp,
                        gpus=args.ray_gpus,
                        max_num_batched_tokens=args.num_generations*args.max_new_tokens+args.max_prefill_len,
                        max_model_len =args.max_new_tokens+args.max_prefill_len)
    import uvicorn
    uvicorn.run(app, host=ray_ip, port=8000)
        # import time
        # while True:
            # time.sleep(3600)
    # except KeyboardInterrupt:
        # ray.shutdown()
