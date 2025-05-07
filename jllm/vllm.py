from jllm.cat2hf import parallel_type

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
                        tensor=tensor.tile(self.tp_size,1)
                    elif dim==1:
                        tensor=tensor.tile(1,self.tp_size)
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
                        tensor = tensor.tile(self.tp_size,1)
                    elif dim==1:
                        tensor = tensor.tile(1,self.tp_size)
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