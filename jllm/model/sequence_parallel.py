import torch

_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_WORLD_SIZE = None
_SEQUENCE_PARALLEL_GLOBAL_RANKS = None
_SEQUENCE_PARALLEL_RANK = None

REWARD_SCORE = None
def get_reward():
    return REWARD_SCORE

def initialize_sequence_parallel(data_parallel_size, pipeline_model_parallel_size, tensor_model_parallel_size,sequence_parallel_size):
    data_parallel_size = data_parallel_size//sequence_parallel_size
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    groups = torch.LongTensor(range(world_size)).reshape(data_parallel_size,sequence_parallel_size, pipeline_model_parallel_size, tensor_model_parallel_size)
    i,t,j,k = torch.where(groups == rank)
    I,T,J,K = i[0],t[0],j[0],k[0]
    global _SEQUENCE_PARALLEL_GROUP, _SEQUENCE_PARALLEL_GLOBAL_RANKS
    for i in range(data_parallel_size):
        for j in range(pipeline_model_parallel_size):
            for k in range(tensor_model_parallel_size):
                ranks = groups[i,:,j,k].tolist()
                group = torch.distributed.new_group(ranks)
                if (i,j,k)==(I,J,K):
                    _SEQUENCE_PARALLEL_GROUP = group
                    _SEQUENCE_PARALLEL_GLOBAL_RANKS = ranks
                    
def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_PARALLEL_GROUP is not None, \
        'sequence parallel group is not initialized'
    return _SEQUENCE_PARALLEL_GROUP
    
def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_WORLD_SIZE
    if _SEQUENCE_PARALLEL_WORLD_SIZE is not None:
        return _SEQUENCE_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())
    
def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_RANK
    if _SEQUENCE_PARALLEL_RANK is not None:
        return _SEQUENCE_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_sequence_parallel_group())
    
def get_sequence_parallel_global_ranks():
    assert _SEQUENCE_PARALLEL_GLOBAL_RANKS is not None, "Sequence parallel group is not initialized"
    return _SEQUENCE_PARALLEL_GLOBAL_RANKS
    
# def get_sequence_parallel_first_rank():
    # assert _SEQUENCE_PARALLEL_GLOBAL_RANKS is not None, "Sequence parallel group is not initialized"
    # return _SEQUENCE_PARALLEL_GLOBAL_RANKS[0]

# def get_sequence_parallel_last_rank():
    # assert _SEQUENCE_PARALLEL_GLOBAL_RANKS is not None, "Sequence parallel group is not initialized"
    # return _SEQUENCE_PARALLEL_GLOBAL_RANKS[-1]

# def get_sequence_parallel_next_rank():
    # assert _SEQUENCE_PARALLEL_GLOBAL_RANKS is not None, "Sequence parallel group is not initialized"
    # rank_in_sequence = get_sequence_parallel_rank()
    # world_size = get_sequence_parallel_world_size()
    # return _SEQUENCE_PARALLEL_GLOBAL_RANKS[(rank_in_sequence + 1) % world_size]

# def get_sequence_parallel_prev_rank():
    # assert _SEQUENCE_PARALLEL_GLOBAL_RANKS is not None, "Sequence parallel group is not initialized"
    # rank_in_sequence = get_sequence_parallel_rank()
    # world_size = get_sequence_parallel_world_size()
    # return _SEQUENCE_PARALLEL_GLOBAL_RANKS[(rank_in_sequence - 1) % world_size]