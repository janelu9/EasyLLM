import sys
import argparse
from functools import wraps
import torch
from torch.distributed import all_gather_into_tensor, reduce_scatter_tensor
from torch_npu.contrib import transfer_to_npu
from mindspeed.arguments import process_args

_ARGS = None


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
    # Need replace modules before import megatron
    aspm.register_patch('importlib.metadata.version', version_wrapper)
    aspm.register_patch('transformer_engine.pytorch.LayerNormLinear', torch.nn.Module, create_dummy=True)
    aspm.register_patch('transformer_engine.pytorch.DotProductAttention', torch.nn.Module, create_dummy=True)
    aspm.register_patch('transformer_engine.pytorch.Linear', torch.nn.Module, create_dummy=True)
    aspm.register_patch('flash_attn.flash_attn_interface.flash_attn_unpadded_func', create_dummy=True)


def apex_adaptation(aspm):
    from mindspeed.optimizer.adamw import AdamW
    from mindspeed.core.fusions.fused_layer_norm import fused_layer_norm_affine
    aspm.register_patch('apex.optimizers.FusedAdam', AdamW, create_dummy=True)
    aspm.register_patch('amp_C.multi_tensor_l2norm', multi_tensor_l2norm, create_dummy=True)
    aspm.register_patch('amp_C.multi_tensor_scale', multi_tensor_scale, create_dummy=True)
    aspm.register_patch('fused_layer_norm_cuda', create_dummy=True)
    aspm.register_patch('apex.multi_tensor_apply.multi_tensor_applier', multi_tensor_applier, create_dummy=True)
    aspm.register_patch('apex.normalization.fused_layer_norm.fused_layer_norm_affine', fused_layer_norm_affine,
                        create_dummy=True)


def torch_adaptation(aspm):
    aspm.register_patch('torch.nn.parameter.Parameter.type', type_wrapper)
    aspm.register_patch('torch.Tensor.type', type_wrapper)
    aspm.register_patch('torch.Tensor.view', ensure_contiguous_wrapper)
    aspm.register_patch('torch.distributed._all_gather_base', all_gather_into_tensor)
    aspm.register_patch('torch.distributed._reduce_scatter_base', reduce_scatter_tensor)


def mcore_models_adaptation_l0(aspm):
    from mindspeed.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec_wrapper
    # Replace FusedLayerNorm with MindSpeed's PTNorm operator in get_gpt-layer
    aspm.register_patch('megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec',
                        get_gpt_layer_local_spec_wrapper)


def mcore_models_adaptation(aspm, mindspeed_args):
    import megatron.core
    megatron.core.jit.jit_fuser = dummy_jit

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from mindspeed.core.models.common.embeddings.rotary_pos_embedding import get_pos_emb_on_this_cp_rank, \
        rotary_embedding_init_wrapper
    aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.get_pos_emb_on_this_cp_rank',
                        get_pos_emb_on_this_cp_rank)
    aspm.register_patch('megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec',
                        get_gpt_layer_local_spec)
    aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.__init__',
                        rotary_embedding_init_wrapper)

    if not mindspeed_args.automated_pipeline and mindspeed_args.noop_layers:
        from mindspeed.core.transformer.transformer_block import _build_layers
        from megatron.core.transformer.transformer_block import TransformerBlock
        from mindspeed.training import num_floating_point_wrapper
        TransformerBlock._build_layers = _build_layers
        aspm.register_patch('megatron.training.training.num_floating_point_operations', num_floating_point_wrapper)

    if mindspeed_args.recompute_norm:
        from mindspeed.core.models.gpt.gpt_layer_specs import build_norm_recompute_layer_wrapper
        aspm.register_patch('megatron.core.transformer.transformer_block.TransformerBlock._build_layers', build_norm_recompute_layer_wrapper)


def mcore_transformer_adaptation_l0(aspm):
    from mindspeed.core.transformer.custom_layers.transformer_engine import PTNorm
    from mindspeed.core.transformer.dot_product_attention import dot_product_attention_forward_wrapper, \
        dot_product_attention_init_wrapper
    aspm.register_patch('megatron.core.transformer.custom_layers.transformer_engine.TENorm', PTNorm)
    # Add cp parameters to dot_deduct_mattention init, and add fusion attention support for alibi in non cp situations
    aspm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.__init__',
                        dot_product_attention_init_wrapper)
    aspm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.forward',
                        dot_product_attention_forward_wrapper)


def mcore_transformer_adaptation(aspm):
    from mindspeed.core.transformer.module import megatron_module_init_wrapper
    from mindspeed.core.transformer.attention import (attention_init_wrapper, SelfAttentionSubmodules,
                                             self_attention_init_wrapper, attention_forward_wrapper)
    from mindspeed.core.transformer.transformer_block import transformer_block_checkpointed_forward_wrapper
    from mindspeed.core.transformer.transformer import parallel_transformer_layer_init_wrapper
    from mindspeed.core.transformer.transformer import core_mlp_forward_wrapper
    aspm.register_patch('megatron.core.transformer.attention.SelfAttentionSubmodules', SelfAttentionSubmodules)
    aspm.register_patch("megatron.core.transformer.attention.Attention.forward", attention_forward_wrapper)
    aspm.register_patch('megatron.core.transformer.attention.Attention.__init__', attention_init_wrapper)
    aspm.register_patch('megatron.core.transformer.attention.SelfAttention.__init__', self_attention_init_wrapper)
    aspm.register_patch('megatron.core.transformer.module.MegatronModule.__init__', megatron_module_init_wrapper)
    aspm.register_patch('megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward',
                        transformer_block_checkpointed_forward_wrapper)
    aspm.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer.__init__',
                        parallel_transformer_layer_init_wrapper)
    aspm.register_patch('megatron.core.transformer.mlp.MLP.forward',
                        core_mlp_forward_wrapper)


def mcore_parallel_state_adaptation(aspm):
    import megatron.core
    from mindspeed.core.parallel_state import initialize_model_parallel_wrapper, initialize_model_parallel
    from mindspeed.core.parallel_state import destroy_model_parallel_wrapper
    from mindspeed.core.memory.auto_pipeline.autopipeline_solver import destroy_model_parallel_profiling_wrapper
    from mindspeed.core.parallel_state import get_context_parallel_group_for_send_recv_overlap
    aspm.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                        initialize_model_parallel_wrapper)
    aspm.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                        initialize_model_parallel)
    aspm.register_patch('megatron.core.parallel_state.destroy_model_parallel',
                        destroy_model_parallel_wrapper)
    aspm.register_patch('megatron.core.parallel_state.destroy_model_parallel',
                        destroy_model_parallel_profiling_wrapper)
    aspm.register_patch('megatron.core.parallel_state.get_context_parallel_group_for_send_recv_overlap',
                        get_context_parallel_group_for_send_recv_overlap)
    aspm.register_patch('megatron.core.mpu', megatron.core.parallel_state)


def mcore_fusions_adaptation(aspm, args):
    from mindspeed.core.fusions.fused_bias_swiglu import SwiGLUFunction, BiasSwiGLUFunction
    from mindspeed.core.fusions.fused_layer_norm import FusedLayerNormAffineFunction, FastLayerNormFN
    from mindspeed.core.fusions.fused_softmax import is_kernel_available, ScaledUpperTriangMaskedSoftmax, ScaledMaskedSoftmax, \
        ScaledSoftmax, forward_fused_softmax
    from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32, npu_matmul_add_fp16
    from mindspeed.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb_bshd
    aspm.register_patch('megatron.core.fusions.fused_layer_norm.FusedLayerNormAffineFunction',
                        FusedLayerNormAffineFunction)
    aspm.register_patch('megatron.core.fusions.fused_layer_norm.FastLayerNormFN', FastLayerNormFN)
    aspm.register_patch('megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax',
                        ScaledUpperTriangMaskedSoftmax)
    aspm.register_patch('megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax', ScaledMaskedSoftmax)
    aspm.register_patch('megatron.core.fusions.fused_softmax.ScaledSoftmax', ScaledSoftmax)
    aspm.register_patch('megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available',
                        is_kernel_available)
    aspm.register_patch('megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax',
                        forward_fused_softmax)
    aspm.register_patch('megatron.core.fusions.fused_bias_swiglu.SwiGLUFunction', SwiGLUFunction)
    aspm.register_patch('megatron.core.fusions.fused_bias_swiglu.BiasSwiGLUFunction', BiasSwiGLUFunction)
    aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32', npu_matmul_add_fp32, create_dummy=True)
    aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16', npu_matmul_add_fp16, create_dummy=True)
    aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.apply_rotary_pos_emb_bshd',
                        apply_rotary_pos_emb_bshd)
    if hasattr(args, 'use_fused_moe_token_permute_and_unpermute') and args.use_fused_moe_token_permute_and_unpermute:
        from mindspeed.core.fusions.npu_moe_token_permute import permute_wrapper
        from mindspeed.core.fusions.npu_moe_token_unpermute import unpermute_wrapper
        aspm.register_patch('megatron.core.transformer.moe.moe_utils.permute', permute_wrapper)
        aspm.register_patch('megatron.core.transformer.moe.moe_utils.unpermute', unpermute_wrapper)


def mcore_optimizer_adapation(aspm):
    from mindspeed.optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper
    from mindspeed.optimizer.optimizer import (mixed_precision_optimizer_step,
                                      reuse_fp32_param_init_wrapper, optimizer_config_init_wrapper)
    # optim relative.
    aspm.register_patch('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step',
                        mixed_precision_optimizer_step)
    aspm.register_patch('megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__',
                        reuse_fp32_param_init_wrapper)
    aspm.register_patch('megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__',
                        optimizer_config_init_wrapper)
    aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                        reuse_fp32_param_distrib_optimizer_init_wrapper)


def mcore_pipeline_parallel_adaptation(aspm):
    from mindspeed.core.pipeline_parallel.p2p_communication import _communicate_shapes
    from mindspeed.core.pipeline_parallel.schedules import get_forward_backward_func_wrapper
    from mindspeed.core.performance.auto_pipeline_perf.schedules import get_forward_backward_func_decorator, \
        backward_step_decorator, forward_step_decorator

    aspm.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                        get_forward_backward_func_wrapper)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                        get_forward_backward_func_decorator)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.backward_step',
                        backward_step_decorator)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_step',
                        forward_step_decorator)
    aspm.register_patch('megatron.core.pipeline_parallel.p2p_communication._communicate_shapes',
                        _communicate_shapes)


def mcore_tensor_parallel_adaptation_l0(aspm):
    from mindspeed.core.tensor_parallel.random import _set_cuda_rng_state
    import jllm
    aspm.register_patch('jllm.core.tensor_parallel.random._set_cuda_rng_state', _set_cuda_rng_state)


def mcore_tensor_parallel_adaptation_l1(aspm):
    from mindspeed.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy_forward
    # use logical negation followed by multiplication to achieve the same effect as setting selected elements to zero
    import jllm
    aspm.register_patch('jllm.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
                        vocab_parallel_cross_entropy_forward)


def mcore_tensor_parallel_adaptation(aspm):
    from mindspeed.core.tensor_parallel.random import checkpoint_wrapper
    from mindspeed.core.tensor_parallel.random import checkpoint_function_backward
    from mindspeed.core.tensor_parallel.layers import vocab_parallel_embedding_forward
    from mindspeed.core.tensor_parallel.layers import row_parallel_nocomm_optimizer_wrapper
    from mindspeed.core.tensor_parallel.layers import parallel_linear_init_wrapper
    import jllm 
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


def megatron_core_adaptation(aspm):
    from mindspeed.core.data_parallel.distributed_data_parallel import distributed_data_parallel_init_wrapper
    aspm.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                        distributed_data_parallel_init_wrapper)


def megatron_legacy_adaptation(aspm):
    from mindspeed.model.language_model import parallel_lm_logits, embedding_forward_wrapper
    from mindspeed.core.performance.auto_pipeline_perf.data_samplers import build_pretraining_data_loader_decorator
    from mindspeed.core.performance.auto_pipeline_perf.transformer import get_attention_mask_wrapper
    from mindspeed.model.gpt_model import post_language_model_processing_wrapper
    aspm.register_patch('mindspeed.model.transformer.get_attention_mask', get_attention_mask_wrapper)
    aspm.register_patch('megatron.legacy.data.data_samplers.build_pretraining_data_loader',
                        build_pretraining_data_loader_decorator)
    aspm.register_patch('megatron.legacy.model.language_model.parallel_lm_logits', parallel_lm_logits)
    aspm.register_patch('megatron.legacy.model.language_model.Embedding.forward', embedding_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.gpt_model.post_language_model_processing', post_language_model_processing_wrapper)


def legacy_model_fusions_adaptation(aspm):
    from mindspeed.core.fusions.fused_layer_norm import FusedLayerNormAffineFunction, FastLayerNormFN, fused_layer_norm_affine
    from mindspeed.core.fusions.fused_softmax import is_kernel_available, ScaledUpperTriangMaskedSoftmax, ScaledMaskedSoftmax, \
        ScaledSoftmax, forward_fused_softmax
    aspm.register_patch('megatron.legacy.model.fused_layer_norm.FusedLayerNormAffineFunction',
                        FusedLayerNormAffineFunction)
    aspm.register_patch('megatron.legacy.model.fused_layer_norm.FastLayerNormFN', FastLayerNormFN)
    aspm.register_patch('megatron.legacy.model.fused_layer_norm.fused_layer_norm_affine', fused_layer_norm_affine)
    aspm.register_patch('megatron.legacy.model.fused_softmax.ScaledUpperTriangMaskedSoftmax',
                        ScaledUpperTriangMaskedSoftmax)
    aspm.register_patch('megatron.legacy.model.fused_softmax.ScaledMaskedSoftmax', ScaledMaskedSoftmax)
    aspm.register_patch('megatron.legacy.model.fused_softmax.ScaledSoftmax', ScaledSoftmax)
    aspm.register_patch('megatron.legacy.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available',
                        is_kernel_available)
    aspm.register_patch('megatron.legacy.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax',
                        forward_fused_softmax)


def legacy_model_rms_norm_adaptation(aspm):
    from mindspeed.core.fusions.rms_norm import rms_norm_init_wrapper, rms_norm_forward_wrapper, rms_norm_norm_wrapper
    aspm.register_patch('megatron.legacy.model.rms_norm.RMSNorm.__init__', rms_norm_init_wrapper)
    aspm.register_patch('megatron.legacy.model.rms_norm.RMSNorm.forward', rms_norm_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.rms_norm.RMSNorm._norm', rms_norm_norm_wrapper)


def legacy_model_transformer_l0(aspm):
    from mindspeed.model.transformer import parallel_mlp_init_wrapper, flash_self_attention_forward, \
        flash_self_attention_init_wrapper, parallel_transformer_forward_wrapper
    from mindspeed.model.transformer import parallel_attention_init_wrapper, parallel_attention_forward
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.forward',
                        parallel_transformer_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelMLP.__init__', parallel_mlp_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.FlashSelfAttention.forward', flash_self_attention_forward)
    aspm.register_patch('megatron.legacy.model.transformer.FlashSelfAttention.__init__',
                        flash_self_attention_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelAttention.__init__', parallel_attention_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelAttention.forward',
                        parallel_attention_forward)


def legacy_model_transformer(aspm, args):
    from mindspeed.model.transformer import parallel_mlp_forward_wrapper, parallel_transformer_init_wrapper, \
        parallel_transformer_init
    from mindspeed.model.transformer import core_attention_init_wrapper, core_attention_forward
    from mindspeed.core.transformer.transformer import parallel_transformer_layer_forward_wrapper, \
        parallel_transformer_checkpointed_forward_wrapper
    from mindspeed.model.transformer import switch_mlp_init_wrapper, switch_mlp_forward_wrapper, \
        parallel_transformer_layer_init_wrapper
    if not args.automated_pipeline and args.noop_layers:
        aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.__init__', parallel_transformer_init)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.__init__',
                        parallel_transformer_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelMLP.forward', parallel_mlp_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.CoreAttention.__init__', core_attention_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.CoreAttention.forward', core_attention_forward)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformerLayer.forward',
                        parallel_transformer_layer_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer._checkpointed_forward',
                        parallel_transformer_checkpointed_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.SwitchMLP.__init__', switch_mlp_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.SwitchMLP.forward', switch_mlp_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformerLayer.__init__',
                        parallel_transformer_layer_init_wrapper)


def megatron_training_adaptation_l0(aspm, args):
    from mindspeed.initialize import _compile_dependencies, set_jit_fusion_options_wrapper
    from mindspeed.utils import get_batch_on_this_cp_rank
    from mindspeed.training import pretrain
    from mindspeed.arguments import parse_args_wrapper, validate_args_wrapper, core_transformer_config_from_args_wrapper
    from mindspeed.yaml_arguments import core_transformer_config_from_yaml_wrapper, print_args_wrapper

    from mindspeed.core.training import train_decorator, train_step_decorator, save_checkpoint_and_time_decorator
    if args.enable_high_availability:
        aspm.register_patch('mindspeed.training.train_uce', train_decorator)
    else:
        aspm.register_patch('megatron.training.training.train', train_decorator)

    aspm.register_patch('megatron.training.training.train_step', train_step_decorator)
    aspm.register_patch('megatron.training.training.save_checkpoint_and_time', save_checkpoint_and_time_decorator)
    aspm.register_patch('megatron.training.yaml_arguments.core_transformer_config_from_yaml',
                        core_transformer_config_from_yaml_wrapper)
    aspm.register_patch('megatron.training.initialize._compile_dependencies', _compile_dependencies)
    aspm.register_patch('megatron.training.utils.get_batch_on_this_cp_rank', get_batch_on_this_cp_rank)
    aspm.register_patch('megatron.training.arguments.parse_args', parse_args_wrapper)
    aspm.register_patch('megatron.training.arguments.validate_args', validate_args_wrapper)
    aspm.register_patch('megatron.training.arguments._print_args', print_args_wrapper)
    aspm.register_patch('megatron.training.yaml_arguments.validate_yaml', validate_args_wrapper)
    aspm.register_patch('megatron.training.yaml_arguments._print_args', print_args_wrapper)
    aspm.register_patch('megatron.training.arguments.core_transformer_config_from_args',
                        core_transformer_config_from_args_wrapper)
    aspm.register_patch('megatron.training.initialize.set_jit_fusion_options', set_jit_fusion_options_wrapper)
    aspm.register_patch('megatron.training.training.pretrain', pretrain)


def megatron_training_adaptation(aspm):
    from mindspeed.core.performance.auto_pipeline_perf.global_vars import get_num_microbatches_wrapper
    from mindspeed.utils import get_batch_on_this_tp_rank
    from mindspeed.tokenizer import build_tokenizer_wrapper
    from mindspeed.core.training import pretrain_decorator, setup_model_and_optimizer_decorator
    aspm.register_patch('megatron.training.global_vars.get_num_microbatches', get_num_microbatches_wrapper)
    aspm.register_patch('megatron.training.training.pretrain', pretrain_decorator)
    aspm.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_decorator)
    aspm.register_patch('megatron.training.utils.get_batch_on_this_tp_rank', get_batch_on_this_tp_rank)
    aspm.register_patch('megatron.training.tokenizer.tokenizer.build_tokenizer', build_tokenizer_wrapper)


def ascend_adaptation(aspm, args):
    from megatron.legacy.model.transformer import ParallelTransformerLayer
    if args.memory_fragmentation:
        from mindspeed.core.memory.memory_fragmentation.pluggable_allocator_adpator import change_allocator
        change_allocator()

        from mindspeed.core.memory.memory_fragmentation.memory_recorder import memory_recorder_wrapper
        aspm.register_patch('megatron.training.training.setup_model_and_optimizer', memory_recorder_wrapper)

        from mindspeed.core.memory.memory_fragmentation.malloc_recorder import malloc_recorder_wrapper
        aspm.register_patch('megatron.training.training.train_step', malloc_recorder_wrapper)

        from mindspeed.core.memory.memory_fragmentation.optimizer_init_precise import optimizer_init_wrapper
        aspm.register_patch('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step', optimizer_init_wrapper)

        from mindspeed.core.memory.adaptive_recomputing.adaptive_recompute import allowed_recomputing_module_wrapper
        allowed_recomputing_module_wrapper(ParallelTransformerLayer)
        from mindspeed.core.memory.adaptive_recomputing.adaptive_recompute import setup_model_and_optimizer_wrapper
        aspm.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_wrapper)
    adaptive_recompute_enable = args.adaptive_recompute_device_size > 0 or args.adaptive_recompute_device_swap
    if (adaptive_recompute_enable and not args.memory_fragmentation) or args.swap_attention:
        from mindspeed.core.memory.adaptive_recomputing.pluggable_allocator_adpator import change_allocator
        if not args.swap_attention:
            change_allocator()
        from mindspeed.core.memory.adaptive_recomputing.adaptive_recompute import allowed_recomputing_module_wrapper
        if hasattr(args, "use_mcore_models") and args.use_mcore_models:
            from megatron.core.transformer.transformer_layer import TransformerLayer
            allowed_recomputing_module_wrapper(TransformerLayer)
        else:
            allowed_recomputing_module_wrapper(ParallelTransformerLayer)
        from mindspeed.core.memory.adaptive_recomputing.adaptive_recompute import setup_model_and_optimizer_wrapper
        aspm.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_wrapper)

    if adaptive_recompute_enable or args.memory_fragmentation:
        import megatron.training.initialize
        aspm.register_patch('megatron.training.initialize_megatron', megatron.training.initialize.initialize_megatron)


def mcore_moe_adaptation_l0(pm):
    from mindspeed.core.transformer.moe.grouped_gemm_util import Ops, grouped_gemm_is_available, get_device_capability
    pm.register_patch('megatron.core.transformer.moe.grouped_gemm_util.ops', Ops)
    pm.register_patch('megatron.core.transformer.moe.grouped_gemm_util.grouped_gemm_is_available',
                      grouped_gemm_is_available)
    pm.register_patch('torch.cuda.get_device_capability', get_device_capability)


def mcore_moe_adaptation(pm, args):
    from mindspeed.core.pipeline_parallel.schedules import forward_step
    pm.register_patch('megatron.core.pipeline_parallel.schedules.forward_step',
                        forward_step)
    if args.moe_permutation_async_comm:
        if hasattr(args, 'moe_token_dispatcher_type') and args.moe_token_dispatcher_type == 'alltoall':
            from mindspeed.core.transformer.moe.experts import sequential_mlp_forward
            from mindspeed.core.transformer.moe.moe_utils import permute, unpermute
            if args.moe_tp_extend_ep:
                from mindspeed.core.transformer.moe.token_dispatcher import (
                    preprocess_tp_extend_ep, alltoall_token_unpermutation_tp_extend_ep,
                    alltoall_token_permutation_tp_extend_ep
                )
                from mindspeed.core.transformer.moe.router import routing_tp_extend_ep
                from mindspeed.core.transformer.moe.moe_layer import base_moe_init_wrapper
                pm.register_patch('megatron.core.transformer.moe.moe_layer.BaseMoELayer.__init__',
                                  base_moe_init_wrapper)
                pm.register_patch(
                    'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.preprocess',
                    preprocess_tp_extend_ep)
                pm.register_patch('megatron.core.transformer.moe.router.TopKRouter.routing', routing_tp_extend_ep)

                if args.moe_alltoall_overlap_comm:
                    from mindspeed.core.transformer.moe.token_dispatcher import alltoall_token_permutation_new, \
                        alltoall_token_unpermutation_new
                    from mindspeed.core.transformer.moe.experts import group_mlp_forward
                    pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
                    pm.register_patch(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
                        alltoall_token_permutation_new)
                    pm.register_patch(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
                        alltoall_token_unpermutation_new)
                else:
                    pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
                                      alltoall_token_permutation_tp_extend_ep)
                    pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
                                      alltoall_token_unpermutation_tp_extend_ep)
            else:
                from mindspeed.core.transformer.moe.token_dispatcher import preprocess, alltoall_token_permutation
                pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.preprocess', preprocess)
                if args.moe_alltoall_overlap_comm:
                    from mindspeed.core.transformer.moe.token_dispatcher import preprocess, alltoall_token_permutation
                    from mindspeed.core.transformer.moe.token_dispatcher import alltoall_token_permutation_new, \
                        alltoall_token_unpermutation_new
                    from mindspeed.core.transformer.moe.experts import group_mlp_forward
                    pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
                    pm.register_patch(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
                        alltoall_token_permutation_new)
                    pm.register_patch(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
                        alltoall_token_unpermutation_new)
                else:
                    pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
                                      alltoall_token_permutation)
            pm.register_patch('megatron.core.transformer.moe.experts.SequentialMLP.forward', sequential_mlp_forward)
            pm.register_patch('megatron.core.transformer.moe.moe_utils.permute', permute)
            pm.register_patch('megatron.core.transformer.moe.moe_utils.unpermute', unpermute)
        else:
            from mindspeed.core.transformer.moe.router import aux_loss_load_balancing
            pm.register_patch('megatron.core.transformer.moe.router.TopKRouter.aux_loss_load_balancing', aux_loss_load_balancing)

            if args.moe_tp_extend_ep:
                from mindspeed.core.transformer.moe.moe_layer import base_moe_init_wrapper
                pm.register_patch('megatron.core.transformer.moe.moe_layer.BaseMoELayer.__init__', base_moe_init_wrapper)

            if args.moe_allgather_overlap_comm:
                from mindspeed.core.transformer.moe.token_dispatcher import (allgather_token_permutation_new,
                                                                    allgather_token_unpermutation_new)
                from mindspeed.core.transformer.moe.experts import group_mlp_forward
                from mindspeed.core.transformer.mlp import mlp_init_wrapper
                pm.register_patch('megatron.core.transformer.mlp.MLP.__init__', mlp_init_wrapper)
                pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
                pm.register_patch(
                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_permutation',
                    allgather_token_permutation_new)
                pm.register_patch(
                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_unpermutation',
                    allgather_token_unpermutation_new)
            else:
                from mindspeed.core.transformer.moe.token_dispatcher import (allgather_token_permutation,
                                                                    allgather_token_unpermutation)
                pm.register_patch(
                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_permutation',
                    allgather_token_permutation)
                pm.register_patch(
                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_unpermutation',
                    allgather_token_unpermutation)

        from mindspeed.core.transformer.moe.moe_layer import moe_layer_init_wrapper
        pm.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer.__init__', moe_layer_init_wrapper)

    from mindspeed.core.transformer.moe.experts import groupedmlp_init_wrapper, groupedmlp_forward_wrapper
    pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.__init__', groupedmlp_init_wrapper)
    pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', groupedmlp_forward_wrapper)

    if args.use_ascend_mc2 and not hasattr(args, 'moe_grouped_gemm'):
        # MoE MLP not use mc2 linear
        from mindspeed.core.models.gpt.gpt_layer_specs import build_layers_wrapper
        from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from megatron.core.transformer.transformer_block import TransformerBlock
        TransformerBlock._build_layers = build_layers_wrapper(TransformerBlock._build_layers, ColumnParallelLinear.forward,
            RowParallelLinear.forward)


def deepspeed_moe_adaptation(pm, args):
    if args.use_pipe_experts or args.use_nanopipe:
        from mindspeed.core.tensor_parallel.layers import (row_parallel_moe, column_parallel_moe,
                                                  linear_with_grad_accumulation_and_async_allreduce_moe)
        pm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.forward', row_parallel_moe)
        pm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward', column_parallel_moe)
        pm.register_patch('megatron.core.tensor_parallel.layers.linear_with_grad_accumulation_and_async_allreduce',
                          linear_with_grad_accumulation_and_async_allreduce_moe)
    if args.use_pipe_experts:
        from mindspeed.core.distributed.param_and_grad_buffer import pipe_register_grad_ready
        pm.register_patch('megatron.core.distributed.ParamAndGradBuffer.register_grad_ready', pipe_register_grad_ready)


def coc_adaptation(aspm, args):
    from mindspeed.initialize import coc_registration_wrapper, mc2_wrapper
    if args.use_ascend_mc2:
        from mindspeed.core.memory.auto_pipeline.autopipeline import initialize_cfg_from_args_wrapper
        aspm.register_patch('megatron.training.initialize.initialize_megatron', mc2_wrapper)
        aspm.register_patch('mindspeed.core.tensor_parallel.ascend_turbo.initialize.initialize_cfg_from_args',
                            initialize_cfg_from_args_wrapper)
    if args.use_ascend_coc:
        aspm.register_patch('megatron.training.initialize.initialize_megatron', coc_registration_wrapper)


def zero3_adaptation(aspm, args):
    if args.enable_zero3:
        from mindspeed.core.data_parallel.distributed_data_parallel import distributed_data_parallel_init_zero3, distributed_data_parallel_zero_grad_wrapper
        from mindspeed.core.tensor_parallel.layers import (parallel_linear_init_zero3_wrapper, column_parallel_linear_forward_zero3,
                                                linear_forward_zero3_wrapper, linear_backward_zero3_wrapper,
                                                row_parallel_linear_forward_zero3, linear_with_grad_accumulation_and_async_allreduce_zero3)
        from mindspeed.optimizer.distrib_optimizer import (build_optimizer_group_ranges_zero3_wrapper, _copy_main_params_to_model_params_zero3,
                                                _copy_model_grads_to_main_grads_zero3, build_model_and_main_param_groups_zero3_wrapper,
                                                distributed_optimizer_zero3_init)
        aspm.register_patch('megatron.core.tensor_parallel.layers.linear_with_grad_accumulation_and_async_allreduce',
                            linear_with_grad_accumulation_and_async_allreduce_zero3)
        aspm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.__init__',
                            parallel_linear_init_zero3_wrapper)
        aspm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.__init__',
                            parallel_linear_init_zero3_wrapper)
        aspm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward',
                            column_parallel_linear_forward_zero3)
        aspm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.forward',
                            row_parallel_linear_forward_zero3)
        aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._build_optimizer_group_ranges',
                            build_optimizer_group_ranges_zero3_wrapper)
        aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._copy_main_params_to_model_params',
                            _copy_main_params_to_model_params_zero3)
        aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._copy_model_grads_to_main_grads',
                            _copy_model_grads_to_main_grads_zero3)
        aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._build_model_and_main_param_groups',
                            build_model_and_main_param_groups_zero3_wrapper)
        aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                            distributed_optimizer_zero3_init)
        aspm.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.forward',
                            linear_forward_zero3_wrapper)
        aspm.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
                            linear_backward_zero3_wrapper)
        aspm.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                            distributed_data_parallel_init_zero3)
        aspm.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.zero_grad_buffer',
                            distributed_data_parallel_zero_grad_wrapper)


def high_availability_adaptation(aspm, args):
    if args.enable_high_availability:
        from mindspeed.core.data_parallel.distributed_data_parallel import distributed_data_parallel_init_uce
        from mindspeed.core.distributed.param_and_grad_buffer import start_grad_sync_wrapper
        from mindspeed.optimizer.optimizer_init import get_megatron_optimizer, get_megatron_optimizer_based_on_param_groups
        from mindspeed.optimizer.clip_grads import clip_grad_norm_fp32_uce
        from mindspeed.optimizer.distrib_optimizer import distributed_optimizer_uce_init
        from mindspeed.training import train_uce
        aspm.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                            distributed_data_parallel_init_uce)
        aspm.register_patch('megatron.core.distributed.param_and_grad_buffer.Bucket.start_grad_sync',
                            start_grad_sync_wrapper)
        aspm.register_patch('megatron.core.optimizer.get_megatron_optimizer', get_megatron_optimizer)
        aspm.register_patch('megatron.core.optimizer.get_megatron_optimizer_based_on_param_groups', get_megatron_optimizer_based_on_param_groups)
        aspm.register_patch('megatron.core.optimizer.clip_grad_norm_fp32', clip_grad_norm_fp32_uce)
        aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__', distributed_optimizer_uce_init)
        aspm.register_patch('megatron.training.training.train', train_uce)


def adaptation_l0(aspm, args):
    """
    The minimum patch set for megatron to adapt to NPU
    """
    # transformer_engine
    te_adaptation(aspm)
    apex_adaptation(aspm)
    torch_adaptation(aspm)
    # Need replace transformer_engine modules before import megatron
    aspm.apply_patches()

    mcore_models_adaptation_l0(aspm)
    mcore_tensor_parallel_adaptation_l0(aspm)
    mcore_transformer_adaptation_l0(aspm)
    mcore_moe_adaptation_l0(aspm)
    legacy_model_transformer_l0(aspm)
    megatron_training_adaptation_l0(aspm, args)
    # context parallel(ring attention) requires mcore parallel state patch
    mcore_parallel_state_adaptation(aspm)


def adaptation_l1(aspm, mindspeed_args):
    """
    Affinity optimization (fusion operators, etc.)
    """
    # fusion operators
    mcore_fusions_adaptation(aspm, mindspeed_args)
    legacy_model_fusions_adaptation(aspm)
    # affinity optimization
    mcore_tensor_parallel_adaptation_l1(aspm)


def adaptation_l2(aspm, mindspeed_args):
    """
    Advanced acceleration algorithm
    """
    megatron_core_adaptation(aspm)
    mcore_models_adaptation(aspm, mindspeed_args)
    mcore_optimizer_adapation(aspm)
    mcore_pipeline_parallel_adaptation(aspm)
    mcore_tensor_parallel_adaptation(aspm)
    mcore_transformer_adaptation(aspm)

    # megatron legacy
    megatron_legacy_adaptation(aspm)
    legacy_model_transformer(aspm, mindspeed_args)
    legacy_model_rms_norm_adaptation(aspm)

    megatron_training_adaptation(aspm)
    ascend_adaptation(aspm, mindspeed_args)
    coc_adaptation(aspm, mindspeed_args)
    mcore_moe_adaptation(aspm, mindspeed_args)
    deepspeed_moe_adaptation(aspm, mindspeed_args)
    zero3_adaptation(aspm, mindspeed_args)
    high_availability_adaptation(aspm, mindspeed_args)


def exe_adaptation():
    mindspeed_args = get_mindspeed_args()
    from mindspeed.patch_utils import MindSpeedPatchesManager as aspm

    if mindspeed_args.optimization_level >= 0:
        # The minimum patch set for megatron to adapt to NPU
        adaptation_l0(aspm, mindspeed_args)

    if mindspeed_args.optimization_level >= 1:
        # Affinity optimization (fusion operators, etc.)
        adaptation_l1(aspm, mindspeed_args)

    if mindspeed_args.optimization_level >= 2:
        # Advanced acceleration algorithm
        adaptation_l2(aspm, mindspeed_args)

    aspm.apply_patches()

    # accelerate package will check TE on sys.modules，so we need remove this patch
    del sys.modules['transformer_engine']
    
    from jllm.core.tensor_parallel.mappings import (
        copy_to_tensor_model_parallel_region,
        gather_from_tensor_model_parallel_region,
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
        scatter_to_tensor_model_parallel_region,
        _reduce_scatter_along_first_dim,
        _gather_along_first_dim,
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
    )
    from jllm.train_pipe import get_args
    from jllm.core import parallel_state
    from mindspeed.core.tensor_parallel import layers
    from mindspeed.core.tensor_parallel import random
    from mindspeed.core.tensor_parallel import cross_entropy
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
    cross_entropy.get_tensor_model_parallel_group=get_tensor_model_parallel_group
    cross_entropy.get_tensor_model_parallel_rank=get_tensor_model_parallel_rank
    cross_entropy.get_tensor_model_parallel_world_size=get_tensor_model_parallel_world_size
    
    args=get_args()
    args.optimize_recomp_communication_status = 1
    args.optimize_recomp_communication_level = 1
    args.recompute_num_layers = 0
    args.swap_attention = False
    args.sequence_parallel = False
    args.use_ascend_mc2 = False
    
exe_adaptation()
