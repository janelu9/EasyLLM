from typing import Callable
import torch.nn.functional as F
from jllm.core.tensor_parallel.mappings import reduce_scatter_to_sequence_parallel_region,reduce_from_tensor_model_parallel_region
from torch_npu.npu import _lazy_call, device as device_ctx_manager
import torch
from torch import _C

def vocab_parallel_embedding_forward_impl(
        self,
        input_):
            
    if self.tensor_model_parallel_size > 1:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input *= ~input_mask
    else:
        masked_input = input_
        # Get the embeddings.

    if self.deterministic_mode:
        output_parallel = self.weight[masked_input]
    else:
        # F.embedding currently has a non-deterministic backward function
        # For higher accumulation accuracy for bf16 on NPU.
        output_parallel = F.embedding(masked_input, self.weight)

    # Mask the output embedding.
    if self.tensor_model_parallel_size > 1:
        output_parallel *= ~input_mask[..., None]
    # Reduce across all the model parallel GPUs.
    if self.reduce_scatter_embeddings:
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        output_parallel = output_parallel.transpose(0, 1).contiguous()
        output = reduce_scatter_to_sequence_parallel_region(output_parallel)
    else:
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
    return output
    

def _set_cuda_rng_state(new_state, device=-1, graph_safe: bool = False):
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.npu.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)

from jllm.core.tensor_parallel.layers import VocabParallelEmbedding
VocabParallelEmbedding.forward = vocab_parallel_embedding_forward_impl

from jllm.core.tensor_parallel import random
random._set_cuda_rng_state = _set_cuda_rng_state