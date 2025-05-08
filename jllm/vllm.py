import os,asyncio
import ray
from vllm import LLM
from jllm.cat2hf import parallel_type

class vLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)
        self.running = True

    async def run_forever(self):
        while self.running:
            await asyncio.sleep(1)

    def stop(self):
        self.running = False
        
def init_vllm(address,
              model,
              gpu_memory_utilization=0.6,
              tensor_parallel_size=1,
              pipeline_parallel_size=1,
              gpus=1,
              max_num_batched_tokens=1024,
              max_model_len=1024):
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    
    ray.init(address=address,ignore_reinit_error=True)
    pg = placement_group([{"GPU": 1, "CPU": 0}]*gpus)
    ray.get(pg.ready())

    vllm_actor = ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
        ),
    )(vLLM).options(name="llm",namespace="vllm",).remote(
        model=model,
        enforce_eager=True,
        worker_extension_cls="jllm.vllm.WorkerExtension",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        distributed_executor_backend="ray",
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len
    )
    ray.get(vllm_actor.collective_rpc.remote("report_device_id", args=tuple()))
    vllm_actor.run_forever.remote()
    return vllm_actor

class WorkerExtension:

    def report_device_id(self) -> str:
        from vllm.platforms import current_platform
        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        self.tp_size = get_tensor_model_parallel_world_size()
        return self.device_uuid
        
    def report_world_size(self) -> str:
        from vllm.distributed.parallel_state import get_world_group
        return get_world_group().world_size
    
    def report_trainer_rank_in_ray(self, device_uuid):
        from vllm.distributed.parallel_state import get_world_group
        if device_uuid == self.report_device_id():
            return get_world_group().rank
        return -1
    
    def update_weight_by_cpu_handles(self, cpu_handles,tile=True):
        if self.device_uuid in cpu_handles:
            handles=cpu_handles[self.device_uuid]
            weights =[]
            for name, handle in handles:
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

    def init_weight_update_group(self, master_address, master_port, world_size, rank_offset=1, trainer_rank=-1):
        from vllm.distributed.parallel_state import get_world_group
        rank = get_world_group().rank
        if rank != trainer_rank:
            if rank < trainer_rank:
                rank += rank_offset
            self.model_update_group = stateless_init_process_group(
                master_address,
                master_port,
                rank,
                world_size,
                self.device,
            )

    def update_weight_by_nccl(self, name, dtype, shape, trainer_rank=-1):
        from vllm.distributed.parallel_state import get_world_group
        if get_world_group().rank != trainer_rank:
            weight = torch.empty(shape, dtype=dtype, device="cuda")
            self.model_update_group.broadcast(weight,
                                              src=0,
                                              stream=torch.cuda.current_stream())

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

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated
        
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
                        default=3,
                        help="num generations")
    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=64,
                        help="max new tokens")
    parser.add_argument("--ray_ip",
                        type=str,
                        default=None,
                        help="ray's master ip.")
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
                        default=0.6,
                        help="vllm gpu_memory_utilization")
    parser.add_argument("--ray_gpus",
                        type=int,
                        default=1,
                        help="num of each ray cluster's gpus.")
                        
    args=parser.parse_args()
    args.model=os.path.abspath(args.model)
    from vllm.utils import get_ip
    import daemon
    from daemon import pidfile
    ray_ip = get_ip() if args.ray_ip is None else args.ray_ip
    with daemon.DaemonContext(
        pidfile=pidfile.TimeoutPIDLockFile("/tmp/vllm_service.pid"),
        stdout=open('vllm_service.log', 'w+'),
        stderr=open('vllm_service_error.log', 'w+'),
        files_preserve=[],
    ):
        actor=init_vllm(f"{ray_ip}:6380",
                   args.model,
                   gpu_memory_utilization=args.vllm_mem,
                   tensor_parallel_size=args.vllm_tp,
                   pipeline_parallel_size=args.vllm_pp,
                   gpus=args.ray_gpus,
                   max_num_batched_tokens=args.num_generations*args.max_new_tokens+args.max_prefill_len,
                   max_model_len =args.max_new_tokens+args.max_prefill_len)
        ray.get(actor.run_forever.remote()) 
