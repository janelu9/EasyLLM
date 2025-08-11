import os
import torch

from deepspeed.runtime.pipe.module import PipelineModule,logger,ds_utils,LayerSpec,nn
def custom_partition_layers(self, method='uniform'):
    num_stages = self._topo.get_dim('pipe')
    stage_id = self._topo.get_coord(self.global_rank).pipe

    if self.global_rank == 0:
        logger.info(f'Partitioning pipeline stages with method {method}')

    method = method.lower()

    # Each stage gets a simple uniform number of layers.
    if method == 'uniform':
        num_layers = len(self._layer_specs)
        self.parts = ds_utils.partition_uniform(num_items=num_layers, num_parts=num_stages)
    elif method == 'parameters':
        param_counts = self._count_layer_params()
        self.parts = ds_utils.partition_balanced(weights=param_counts, num_parts=num_stages)
    elif method.startswith('type:'):
        layertype = method.split(':')[1]
        binary_weights = [0] * len(self._layer_specs)
        for idx in self._find_layer_type(layertype):
            binary_weights[idx] = 1
        self.parts = ds_utils.partition_balanced(weights=binary_weights, num_parts=num_stages)
    elif method == 'profile':
        raise NotImplementedError(f'Partitioning method {method} not implemented.')
    elif ',' in method:
        self.parts = list(map(int,method.split(',')))
    else:
        raise NotImplementedError(f'Partitioning method {method} not implemented.')

    # Print some information on the partitioning.
    if self.global_rank == 0:
        for stage in range(num_stages):
            start = self.parts[stage]
            stop = self.parts[stage + 1]
            print(f'stage={stage} layers={stop - start}')
            for idx, layer in enumerate(self._layer_specs[start:stop]):
                name = str(layer)
                num_layers = ''
                if isinstance(layer, LayerSpec):
                    name = layer.typename.__name__
                    num_layers = layer.module_kwargs.get('num_layers','')
                if isinstance(layer, nn.Module):
                    name = layer.__class__.__name__
                else:
                    try:
                        name = layer.__name__
                    except AttributeError:
                        pass
                print(f'    {idx+start:2d}: {name} {num_layers}')
        if self.loss_fn:
            try:
                print(f'  loss: {self.loss_fn.__name__}')
            except AttributeError:
                print(f'  loss: {self.loss_fn.__class__.__name__}')

    self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])
PipelineModule._partition_layers = custom_partition_layers

from deepspeed.runtime.state_dict_factory import SDLoaderBase
def check_ckpt_list(self):
    #logger.info(f'checkpoint file list: {self.ckpt_list}')
    assert len(self.ckpt_list) > 0

    sd = self.checkpoint_engine.load(self.ckpt_list[0], map_location=lambda storage, loc: storage)

    # check checkpoint count is same with saved mp_world_size
    # if 'mp_world_size' in sd.keys():
        # assert len(self.ckpt_list) == sd[
            # 'mp_world_size'], f"checkpoint count {len(self.ckpt_list)} is different from saved mp_world_size {sd['mp_world_size']}"
SDLoaderBase.check_ckpt_list = check_ckpt_list

from jllm.train_pipe import args
if args.rlhf:
    from deepspeed.runtime.bf16_optimizer import get_global_norm_of_tensors,get_norm_with_moe_layers,clip_tensors_by_global_norm
    from deepspeed.runtime import bf16_optimizer
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError(f'{self.__class__} does not support closure.')

        non_expert_grads_for_norm, expert_grads_for_norm = self.get_grads_for_norm()
        non_expert_groups_norm = get_global_norm_of_tensors(input_tensors=non_expert_grads_for_norm,
                                                            mpu=self.mpu,
                                                            norm_type=self.norm_type,
                                                            use_graph=self.graph_harvesting)
        all_groups_norm = non_expert_groups_norm
        if self.has_moe_layers:
            all_groups_norm = get_norm_with_moe_layers(non_expert_groups_norm,
                                                       mpu=self.mpu,
                                                       expert_tensors=expert_grads_for_norm,
                                                       norm_type=self.norm_type)

        self._global_grad_norm = all_groups_norm

        assert all_groups_norm >= 0.
        if self.clip_grad > 0.:
            clip_tensors_by_global_norm(input_tensors=self.get_grads_for_norm(for_clipping=True),
                                        max_norm=self.clip_grad,
                                        global_norm=all_groups_norm,
                                        mpu=self.mpu,
                                        use_graph=self.graph_harvesting)

        for param_partition, grad_partition in zip(self.fp32_groups_flat_partition,
                                                   self.fp32_groups_gradient_flat_partition):
            # In case of grad acc dtype different than FP32, need to cast to high precision.
            param_partition.grad = grad_partition.to(
                param_partition.dtype) if grad_partition.dtype != param_partition.dtype else grad_partition

        self.optimizer.step()

        if self.grad_acc_dtype is not torch.float32:
            for param_partition in self.fp32_groups_flat_partition:
                param_partition.grad = None

        # We need to link optimizer state after the first step() call
        self._lazy_init_hp_params_optimizer_state()

        self.update_lp_params()

        self.clear_hp_grads()
    bf16_optimizer.BF16_Optimizer.step = step
    
    counter = -1
    group_step =args.num_generations//args.sequence_parallel_size//args.micro_batch_size
    
    def backward(self, loss, retain_graph=False, update_hp_grads=True, clear_lp_grads=False, **bwd_kwargs):
        """Perform a backward pass and copy the low-precision gradients to the
        high-precision copy.

        We copy/accumulate to the high-precision grads now to prevent accumulating in the
        bf16 grads after successive backward() calls (i.e., grad accumulation steps > 1)

        The low-precision grads are deallocated during this procedure.
        """
        global counter
        counter = (counter + 1) % group_step
        self.clear_lp_grads()
        loss.backward(retain_graph=counter+1<group_step, **bwd_kwargs)

        if update_hp_grads:
            self.update_hp_grads(clear_lp_grads=clear_lp_grads)
            
    bf16_optimizer.BF16_Optimizer.backward = backward
    
    from deepspeed.runtime.pipe.engine import (
        PartitionedTensor,
        BACKWARD_MICRO_TIMER,
        BACKWARD_GLOBAL_TIMER,
        BACKWARD_INNER_MICRO_TIMER,
        BACKWARD_INNER_GLOBAL_TIMER
    )
    
    counter_0 = -1
    
    def _exec_backward_pass(self, buffer_id):
        
        global counter_0
        counter_0 = (counter_0 + 1) % group_step
        
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            super().backward(self.loss)
            self.mem_status('AFTER BWD')
            return

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.wall_clock_breakdown():
            self.timers(BACKWARD_MICRO_TIMER).start()
            self.timers(BACKWARD_GLOBAL_TIMER).start()
            self.timers(BACKWARD_INNER_MICRO_TIMER).start()
            self.timers(BACKWARD_INNER_GLOBAL_TIMER).start()

        # Reconstruct if we previously partitioned the output. We must be
        # careful to also restore the computational graph of the tensors we partitioned.
        if self.is_pipe_partitioned:
            if self.is_grad_partitioned:
                if self.pipe_partition_output_meta_cache is None:
                    self.pipe_partition_output_meta_cache = outputs[0].to('cpu')
                part_output = PartitionedTensor.from_meta(meta=self.pipe_partition_output_meta_cache,
                                                          local_part=outputs[1],
                                                          group=self.grid.get_slice_parallel_group())
                self.pipe_buffers['output_tensors'][buffer_id].data = part_output.full()
                outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[2:])
            else:
                # Already restored from partition
                self.pipe_buffers['output_tensors'][buffer_id].data = outputs[0]
                outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[1:])

        grad_tensors = self.grad_layer
        if self.is_grad_partitioned:
            #print(f'RANK={self.global_rank} BEFORE-BWD restoring grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')
            if self.grad_partition_grad_layer_meta_cache is None:
                self.grad_partition_grad_layer_meta_cache = self.grad_layer[0].to('cpu')
            part_grad = PartitionedTensor.from_meta(meta=self.grad_partition_grad_layer_meta_cache,
                                                    local_part=self.grad_layer[1],
                                                    group=self.grid.get_slice_parallel_group())
            grad_tensors = (part_grad.full(), *grad_tensors[2:])
            part_grad = None
            #print(f'RANK={self.global_rank} BEFORE-BWD restored grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')

        if self.using_bf16_optimizer and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()

        # This handles either a single tensor or tuple of tensors.
        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if t.is_floating_point()]
            assert len(out_tensors) == len(grad_tensors)
            torch.autograd.backward(tensors=out_tensors,retain_graph=counter_0+1<group_step, grad_tensors=grad_tensors)
        else:
            torch.autograd.backward(tensors=(outputs, ),retain_graph=counter_0+1<group_step, grad_tensors=(grad_tensors, ))

        if self.using_bf16_optimizer and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            if not self._config.bfloat16_immediate_grad_update:
                self.optimizer.update_hp_grads(clear_lp_grads=False)

        # Free up the memory from the output of forward()
        self.pipe_buffers['output_tensors'][buffer_id] = None
        self.pipe_buffers['outputs'][buffer_id] = None
        grad_tensors = None

        if self.wall_clock_breakdown():
            self.timers(BACKWARD_INNER_MICRO_TIMER).stop()
            self.timers(BACKWARD_INNER_GLOBAL_TIMER).stop()
            self.timers(BACKWARD_MICRO_TIMER).stop()
            self.timers(BACKWARD_GLOBAL_TIMER).stop()

        self.mem_status('AFTER BWD')
        
    from deepspeed.runtime.pipe import engine
    engine._exec_backward_pass = _exec_backward_pass


if args.expert_parallel_size>1:
    from deepspeed.moe import layer
    from deepspeed.runtime import engine
    from deepspeed.moe import utils 
    from deepspeed.profiling import flops_profiler 
    from jllm.model.deepseek_v3.parallel_deepseek_v3 import DeepseekV3MoE

    layer.MoE=DeepseekV3MoE
    engine.MoE=DeepseekV3MoE
    utils.MoE=DeepseekV3MoE
    flops_profiler.MoE=DeepseekV3MoE
    
    from deepspeed.runtime.utils import get_global_norm_of_tensors,get_accelerator,inf
    from deepspeed.runtime import bf16_optimizer
    
    def get_norm_with_moe_layers(non_expert_norm, mpu, expert_tensors, norm_type=2):
        def to_tensor(v):
            return get_accelerator().FloatTensor([float(v)]).detach()
        group_norms = [non_expert_norm]
        for exp_name, tensors in expert_tensors.items():
            group_norm = get_global_norm_of_tensors(input_tensors=tensors,
                                                    mpu=mpu,
                                                    norm_type=norm_type,
                                                    use_graph=False,
                                                    moe_ep_group=groups._get_expert_parallel_group(exp_name))
            group_norms.append(group_norm)
        group_norms = torch.stack([to_tensor(norm) for norm in group_norms])
        if group_norms.eq(-1).any():
            return -1
        if norm_type == inf:
            total_norm = group_norms.max().item()
        else:
            total_norm = group_norms.pow(norm_type).sum()
            total_norm = total_norm.item()**(1. / norm_type)
            if total_norm == float('inf') or total_norm == -float('inf'):
                total_norm = -1
        return total_norm
    bf16_optimizer.get_norm_with_moe_layers=get_norm_with_moe_layers

    from deepspeed.runtime.engine import (DeepSpeedEngine,logger,groups,defaultdict,re,remove_random_ltd_state_dict,
                                          TorchCheckpointEngine,)
    def _get_non_moe_state_dict(self, full_state_dict):
        for key in list(full_state_dict.keys()):
            if 'experts' in key and 'shared_experts' not in key:
                full_state_dict.pop(key)
        return full_state_dict

    def _save_moe_checkpoint(self, save_dir, tag, client_state={}, exclude_frozen_parameters=False):
        save_path = self._get_ckpt_name(save_dir, tag)
        moe_layer_id = 0
        for n_module, module in self.module.named_modules():
            if isinstance(module, DeepseekV3MoE):  # and deepspeed.comm.get_rank() == 0:
                group_name = module.expert_group_name
                num_local_experts = module.num_local_experts
                expp_rank = groups._get_expert_parallel_rank(group_name)
                exp_dp_rank = groups._get_expert_data_parallel_rank(group_name)
                # print(expp_rank, exp_dp_rank)
                if exp_dp_rank != 0:
                    moe_layer_id += 1
                    continue

                # get all moe parameters
                moe_state_dict = {}
                for n, p in module.state_dict().items():
                    if 'experts' in n and 'shared_experts' not in n:
                        moe_state_dict[n_module + '.' + n] = p
                moe_str_prefix = '.moe.experts.'
                # print(moe_state_dict.keys()) # until now, everything is fine. So the bug happens at next few lines
                # Reorder the moe name rank, so that each checkpoint only has one expert
                experts_state_dict = defaultdict(dict)
                for key in list(moe_state_dict.keys()):
                    m = re.match(f".*{moe_str_prefix}.*", key)

                    local_expert_id = None
                    if not m:
                        logger.warning(f'No expert found in key {key}.')
                    else:
                        local_expert_id = '0'#m.group(1)

                    global_expert_id = expp_rank * \
                        num_local_experts + int(local_expert_id)
                    expert_key = key.replace(f'{moe_str_prefix}{local_expert_id}',
                                             f'{moe_str_prefix}{global_expert_id}')
                    # truncating extra tensor (shared) storage
                    truncated = moe_state_dict.pop(key).clone().detach()
                    experts_state_dict[str(global_expert_id)][expert_key] = truncated

                # let save the moe parameters
                for global_expert_id, expert_state_dict in experts_state_dict.items():
                    # save the moe parameters
                    moe_save_path = self._get_expert_ckpt_name(save_dir, moe_layer_id, global_expert_id, tag, self.mpu)
                    if self.random_ltd_enabled():
                        expert_state_dict = remove_random_ltd_state_dict(expert_state_dict)
                    self.checkpoint_engine.save(expert_state_dict, moe_save_path)
                moe_layer_id += 1

        self._curr_ckpt_path = os.path.join(save_dir, tag)

        largest_group_name = groups._get_max_expert_size_name()
        expp_rank = groups._get_expert_parallel_rank(largest_group_name)
        exp_dp_rank = groups._get_expert_data_parallel_rank(largest_group_name)

        # In the case of E + D parallelism, only the
        # first expert parallel group should save the expert weights
        # since each expert parallel group is a copy of the model's experts
        if exp_dp_rank == 0:
            # Save optimizer states. They are different across each exp parallel rank.
            optimizer_state = {
                'optimizer': self.optimizer.state_dict() if self.optimizer and not self.zero_optimization() else None
            }
            # TODO: why use BufferedWriter not the path
            file_path = self._get_optimizer_ckpt_name(save_dir, tag, expp_rank)
            self.checkpoint_engine.save(optimizer_state, file_path)

        # Load flow uses below saved file for model parameters, RNG and more
        if groups._get_data_parallel_rank() == 0:
            # Get non-moe parameters
            # Classes DeepSpeedEngine and PipelineEngine have different behavior for method module_state_dict.
            # DeepSpeedEngine returns the state dict, where PipelineEngine saves the state dict and returns None.
            # We need to get the state dict, therefore, call to DeepSpeedEngine (base class for PipelineEngine)
            model_state_dict = self._get_non_moe_state_dict(
                DeepSpeedEngine.module_state_dict(self, exclude_frozen_parameters=exclude_frozen_parameters))

            # TODO: update num experts info,.. in checkpoint
            state = {
                'module':
                model_state_dict,
                'lr_scheduler':
                self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                'data_sampler':
                self.training_dataloader.data_sampler.state_dict() if
                (self.training_dataloader is not None and self.curriculum_learning_enabled()) else None,
                'random_ltd':
                self.random_ltd_scheduler.state_dict() if self.random_ltd_enabled() else None,
                'sparse_tensor_module_names':
                self.sparse_tensor_module_names,
                'skipped_steps':
                self.skipped_steps,
                'global_steps':
                self.global_steps,
                'global_samples':
                self.global_samples,
                'dp_world_size':
                self.dp_world_size,
                'mp_world_size':
                self.mp_world_size,
                'num_experts':
                self.num_experts,
                'ds_config':
                self.config
            }
            state.update(client_state)
            logger.info(f'Saving model checkpoint: {save_path}')
            self.checkpoint_engine.save(state, save_path)
            
    @staticmethod   
    def load_moe_state_dict(checkpoint_path,
                            tag,
                            state_dict,
                            old_moe_load,
                            model=None,
                            mpu=None,
                            num_experts=1,
                            checkpoint_engine=TorchCheckpointEngine()):
        if old_moe_load:
            expp_rank = groups._get_expert_data_parallel_rank(groups._get_max_expert_size_name())

            num_local_experts = max(num_experts) // groups._get_expert_parallel_world_size(
                groups._get_max_expert_size_name())
            for local_expert_id in range(num_local_experts):
                global_expert_id = expp_rank * num_local_experts + local_expert_id
                expert_state_dict = checkpoint_engine.load(
                    DeepSpeedEngine._get_expert_ckpt_name(
                        checkpoint_path,
                        -1,  # -1 means ignore layer_id
                        global_expert_id,
                        tag,
                        mpu),
                    map_location=torch.device('cpu'))

                # Updating global -> local expert ids
                moe_str_prefix = '.moe.experts.'
                for key in list(expert_state_dict.keys()):
                    local_key = key.replace(f'{moe_str_prefix}{global_expert_id}',
                                            f'{moe_str_prefix}{local_expert_id}')
                    expert_state_dict[local_key] = expert_state_dict.pop(key)
                state_dict.update(expert_state_dict)

        else:
            moe_layer_id = 0
            for n_module, module in model.named_modules():
                if isinstance(module, DeepseekV3MoE):  # and deepspeed.comm.get_rank() == 0:
                    group_name = module.expert_group_name
                    num_local_experts = module.num_local_experts
                    expp_rank = groups._get_expert_parallel_rank(group_name)
                    # loop all local_experts
                    for local_expert_id in range(num_local_experts):
                        global_expert_id = expp_rank * num_local_experts + local_expert_id
                        expert_state_dict = checkpoint_engine.load(DeepSpeedEngine._get_expert_ckpt_name(
                            checkpoint_path, moe_layer_id, global_expert_id, tag, mpu),
                                                                   map_location=torch.device('cpu'))
                        # print(expert_state_dict.keys())
                        # Updating global -> local expert ids
                        moe_str_prefix = '.moe.experts.'
                        for key in list(expert_state_dict.keys()):
                            local_key = key.replace(f'{moe_str_prefix}{global_expert_id}',
                                                    f'{moe_str_prefix}{local_expert_id}')
                            expert_state_dict[local_key] = expert_state_dict.pop(key)
                        state_dict.update(expert_state_dict)
                    moe_layer_id += 1
            
    DeepSpeedEngine._get_non_moe_state_dict=_get_non_moe_state_dict
    DeepSpeedEngine._save_moe_checkpoint=_save_moe_checkpoint
    DeepSpeedEngine.load_moe_state_dict=load_moe_state_dict
