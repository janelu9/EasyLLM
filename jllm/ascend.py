import os
import sys
import shutil
import argparse
import time
from functools import wraps
import torch
from torch.distributed import all_gather_into_tensor, reduce_scatter_tensor
from torch_npu.contrib import transfer_to_npu
from multiprocessing import Lock

def add_args(args, key, value):
    if key is not None:
        key = key[2:].replace('-', '_')
        if value is None:
            value = True
        elif len(value) == 1:
            value = value[0]
        setattr(args, key, value)


def parser_unknown_args(args, unknown):
    i = 0
    key = value = None
    while i < len(unknown):
        if unknown[i].startswith("--"):
            add_args(args, key, value)
            key = unknown[i]
            value = None
        else:
            if value is None:
                value = [unknown[i]]
            else:
                value.append(unknown[i])
        i += 1
    add_args(args, key, value)


def get_mindspeed_args():
    global _ARGS
    if _ARGS is None:
        parser = argparse.ArgumentParser(description='MindSpeed Arguments', allow_abbrev=False)
        _ARGS, unknown = process_args(parser).parse_known_args()
        parser_unknown_args(_ARGS, unknown)
    return _ARGS


def dummy_jit(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def lcm(a, b):
    import math
    return (a * b) // math.gcd(a, b)


def type_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if isinstance(res, str):
            res = res.replace('npu', 'cuda')
        return res

    return wrapper


def version_wrapper(fn):
    @wraps(fn)
    def wrapper(name, *args, **kwargs):
        if name == 'transformer-engine':
            return '0.0'
        res = fn(name, *args, **kwargs)
        return res

    return wrapper


# Patch view method to ensure tensor is contiguous before performing view
def ensure_contiguous_wrapper(fn):
    def wrapper(tensor, *args, **kwargs):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return fn(tensor, *args, **kwargs)

    return wrapper


def multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
    return op(noop_flag_buffer, tensor_lists, *args)


def multi_tensor_l2norm(overflow_buf, tensor_lists, per_parameter):
    total_norm = 0.0
    norm_type = 2.0
    ret_per_tensor = [] if per_parameter else None
    for grads_for_norm in tensor_lists:
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += grad_norm ** norm_type
        if per_parameter:
            ret_per_tensor.append(total_norm.clone())
    if not tensor_lists:
        grad_norm = torch.cuda.FloatTensor([0])
        total_norm = grad_norm ** norm_type
    return total_norm ** (1 / norm_type), ret_per_tensor


def multi_tensor_scale(overflow_buf, tensor_lists, scale):
    if len(tensor_lists) != 2:
        raise AssertionError('The size of tensor list must be 2, but got {}'.format(len(tensor_lists)))
    if len(tensor_lists[0]) != len(tensor_lists[1]):
        raise AssertionError('The size of tensor list must be same, but got {} and {}'.format(len(tensor_lists[0]),
                                                                                              len(tensor_lists[1])))

    with torch.no_grad():
        for i in range(len(tensor_lists[0])):
            tensor_lists[1][i].copy_(tensor_lists[0][i] * scale)


def te_adaptation(aspm):
    aspm.register_patch('torch.compile', torch.jit.script)
    # Need replace modules before import jllm
    aspm.register_patch('importlib.metadata.version', version_wrapper)
    aspm.register_patch('transformer_engine.pytorch.LayerNormLinear', torch.nn.Module, create_dummy=True)
    aspm.register_patch('transformer_engine.pytorch.DotProductAttention', torch.nn.Module, create_dummy=True)
    aspm.register_patch('transformer_engine.pytorch.Linear', torch.nn.Module, create_dummy=True)
    aspm.register_patch('transformer_engine.common.recipe.DelayedScaling', torch.nn.Module, create_dummy=True)
    aspm.register_patch('flash_attn.flash_attn_interface.flash_attn_unpadded_func', create_dummy=True)


def apex_adaptation(aspm):
    from mindspeed.core.fusions.fused_layer_norm import fused_layer_norm_affine
    from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32, npu_matmul_add_fp16
    aspm.register_patch('amp_C.multi_tensor_l2norm', multi_tensor_l2norm, create_dummy=True)
    aspm.register_patch('amp_C.multi_tensor_scale', multi_tensor_scale, create_dummy=True)
    aspm.register_patch('fused_layer_norm_cuda', create_dummy=True)
    aspm.register_patch('apex.multi_tensor_apply.multi_tensor_applier', multi_tensor_applier, create_dummy=True)
    aspm.register_patch('apex.normalization.fused_layer_norm.fused_layer_norm_affine', fused_layer_norm_affine,
                        create_dummy=True)
    aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32', npu_matmul_add_fp32, create_dummy=True)
    aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16', npu_matmul_add_fp16, create_dummy=True)


def torch_adaptation(aspm):
    aspm.register_patch('torch.nn.parameter.Parameter.type', type_wrapper)
    aspm.register_patch('torch.Tensor.type', type_wrapper)
    aspm.register_patch('torch.Tensor.view', ensure_contiguous_wrapper)
    aspm.register_patch('torch.distributed._all_gather_base', all_gather_into_tensor)
    aspm.register_patch('torch.distributed._reduce_scatter_base', reduce_scatter_tensor)
    # lmc is supported python >=3.9
    if sys.version_info < (3, 9):
        aspm.register_patch('math.lcm', lcm, create_dummy=True)


def mcore_tensor_parallel_adaptation_l0(aspm):
    from mindspeed.core.tensor_parallel.random import _set_cuda_rng_state
    aspm.register_patch('jllm.core.tensor_parallel.random._set_cuda_rng_state', _set_cuda_rng_state)


def mcore_tensor_parallel_adaptation_l1(aspm):
    from mindspeed.core.tensor_parallel.cross_entropy import calculate_predicted_logits
    # use logical negation followed by multiplication to achieve the same effect as setting selected elements to zero
    aspm.register_patch('jllm.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.calculate_predicted_logits',
                        calculate_predicted_logits)


def mcore_tensor_parallel_adaptation(aspm):
    from mindspeed.core.tensor_parallel.random import checkpoint_wrapper
    from mindspeed.core.tensor_parallel.random import checkpoint_function_backward
    from mindspeed.core.tensor_parallel.layers import vocab_parallel_embedding_forward
    from mindspeed.core.tensor_parallel.layers import row_parallel_nocomm_optimizer_wrapper
    from mindspeed.core.tensor_parallel.layers import parallel_linear_init_wrapper


    aspm.register_patch('jllm.core.tensor_parallel.random.CheckpointFunction.backward',
                        checkpoint_function_backward)
    aspm.register_patch('jllm.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                        vocab_parallel_embedding_forward)
    aspm.register_patch('jllm.core.tensor_parallel.layers.RowParallelLinear.forward',
                        row_parallel_nocomm_optimizer_wrapper)
    aspm.register_patch('jllm.core.tensor_parallel.layers.RowParallelLinear.__init__',
                        parallel_linear_init_wrapper)
    aspm.register_patch('jllm.core.tensor_parallel.layers.ColumnParallelLinear.__init__',
                        parallel_linear_init_wrapper)
    aspm.register_patch('jllm.core.tensor_parallel.random.checkpoint', checkpoint_wrapper)


def delete_lock_file(directory, lock):
    with lock:
        flag_lock = False
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for name in files:
                    if name.endswith('.lock') or name.endswith('lock'):
                        if os.path.exists(directory):
                            flag_lock = True
                            print(f"Process (PID: {os.getpid()}) is deleting Lock directory")
                            shutil.rmtree(directory)
                            print(f"Process (PID: {os.getpid()}) deleted Lock directory")
                            if flag_lock:
                                break
                        else:
                            print(f"Process (PID: {os.getpid()}) Directory {directory} does not exist.")
                if flag_lock:
                    break
                    
def exe_adaptation():
    from torch.utils.cpp_extension import _get_build_directory
    build_directory = _get_build_directory("", True)
    delete_lock = Lock()
    delete_lock_file(build_directory, delete_lock)
    
    from mindspeed.patch_utils import MindSpeedPatchesManager as aspm
    # transformer_engine
    te_adaptation(aspm)
    apex_adaptation(aspm)
    torch_adaptation(aspm)
    # Need replace transformer_engine modules before import jllm
    aspm.apply_patches()
    mcore_tensor_parallel_adaptation_l0(aspm)
    mcore_tensor_parallel_adaptation_l1(aspm)
    mcore_tensor_parallel_adaptation(aspm)
    aspm.apply_patches()
    del sys.modules['transformer_engine']
    from jllm.core.tensor_parallel.mappings import (
        copy_to_tensor_model_parallel_region,
        gather_from_tensor_model_parallel_region,
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
        scatter_to_tensor_model_parallel_region,
        _reduce_scatter_along_first_dim,
        _gather_along_first_dim,
        _reduce
    )
    from jllm.core.tensor_parallel.layers import (
        linear_with_grad_accumulation_and_async_allreduce,
        linear_with_frozen_weight,
    )
    from jllm.core.tensor_parallel.utils import gather_split_1d_tensor
    from jllm.core.tensor_parallel.random import (
        get_cuda_rng_tracker,
        safely_set_viewless_tensor_data
    )
    from jllm.core.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_world_size,
        get_tensor_model_parallel_rank,
        is_pipeline_last_stage,
        get_virtual_pipeline_model_parallel_rank
    )
    from jllm.train_pipe import get_args
    from jllm.core import parallel_state
    from mindspeed.core.tensor_parallel import layers
    from mindspeed.core.tensor_parallel import random
    from mindspeed.core.tensor_parallel import mapping
    layers.copy_to_tensor_model_parallel_region = copy_to_tensor_model_parallel_region
    layers.gather_from_tensor_model_parallel_region =gather_from_tensor_model_parallel_region
    layers.reduce_from_tensor_model_parallel_region =reduce_from_tensor_model_parallel_region
    layers.reduce_scatter_to_sequence_parallel_region =reduce_scatter_to_sequence_parallel_region
    layers.scatter_to_tensor_model_parallel_region =scatter_to_tensor_model_parallel_region
    layers._reduce_scatter_along_first_dim =_reduce_scatter_along_first_dim
    layers._gather_along_first_dim =_gather_along_first_dim
    layers.linear_with_grad_accumulation_and_async_allreduce =linear_with_grad_accumulation_and_async_allreduce
    layers.linear_with_frozen_weight =linear_with_frozen_weight
    layers.get_args =get_args
    layers.parallel_state =parallel_state
    layers.mpu =parallel_state
    random.get_args = get_args
    random.gather_split_1d_tensor = gather_split_1d_tensor
    random.get_cuda_rng_tracker=get_cuda_rng_tracker
    random.safely_set_viewless_tensor_data=safely_set_viewless_tensor_data
    random.get_tensor_model_parallel_group=get_tensor_model_parallel_group
    random.get_tensor_model_parallel_world_size=get_tensor_model_parallel_world_size
    random.is_pipeline_last_stage = is_pipeline_last_stage
    random.get_virtual_pipeline_model_parallel_rank = get_virtual_pipeline_model_parallel_rank
    mapping._reduce=_reduce
    
    args=get_args()
    args.optimize_recomp_communication_status = 0
    args.optimize_recomp_communication_level = 0
    args.recompute_num_layers = 0
    args.swap_attention = False
    args.sequence_parallel = False
    args.use_ascend_mc2 = False
    args.adaptive_memory_optimization = False
    args.op_cal_tflops = False
exe_adaptation()