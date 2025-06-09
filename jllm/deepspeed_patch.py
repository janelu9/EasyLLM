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

from jllm.train_pipe import get_args
if get_args().expert_parallel_size>1:
    from deepspeed.moe import layer
    from deepspeed.runtime import engine,utils
    from deepspeed.moe import utils 
    from deepspeed.profiling import flops_profiler 
    from jllm.model.deepseek_v3.parallel_deepseek_v3 import DeepseekV3MoE

    layer.MoE=DeepseekV3MoE
    engine.MoE=DeepseekV3MoE
    utils.MoE=DeepseekV3MoE
    flops_profiler.MoE=DeepseekV3MoE

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
    utils.get_norm_with_moe_layers=get_norm_with_moe_layers

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
