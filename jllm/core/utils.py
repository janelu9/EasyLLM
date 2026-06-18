# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch
import operator
from packaging.version import Version as PkgVersion
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Iterable
from dataclasses import dataclass, field, replace
from abc import ABC, abstractmethod
import numpy as np

try:
    _torch_version = PkgVersion(torch.__version__)
except Exception:
    # This is a WAR for building docs, where torch is not actually imported
    _torch_version = PkgVersion("0.0.0")

def get_te_version():
    """Get TE version from __version__; if not available use pip's. Use caching."""

    def get_te_version_str():
        import transformer_engine as te

        if hasattr(te, '__version__'):
            return str(te.__version__)
        else:
            return version("transformer-engine")

    global _te_version
    if _te_version is None:
        _te_version = PkgVersion(get_te_version_str())
    return _te_version


def is_te_min_version(version, check_equality=True):
    """Check if minimum version of `transformer-engine` is installed."""
    if check_equality:
        return get_te_version() >= PkgVersion(version)
    return get_te_version() > PkgVersion(version)

  
def get_torch_version():
    """Get torch version from __version__."""

    global _torch_version
    return _torch_version


def is_torch_min_version(version, check_equality=True):
    """Check if minimum version of `torch` is installed."""
    if check_equality:
        return get_torch_version() >= PkgVersion(version)
    return get_torch_version() > PkgVersion(version)
    
class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        """
        Returns (potentially) a sub-tensor from the self.buffer for the given shape.
        """
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)
        
def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator
    
def assert_viewless_tensor(tensor, extra_msg=None):
    """Assert that a tensor is not a view (i.e., its '._base' field is
    not set)."""
    if isinstance(tensor, list):
        [assert_viewless_tensor(t) for t in tensor]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        "Ensure tensor._base is None before setting tensor.data or storing "
        "tensor to memory buffer. Otherwise, a memory leak will occur (and "
        f"likely accumulate over iterations). {extra_msg}"
    )
    return tensor


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    """Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    """
    assert_viewless_tensor(
        tensor,
        extra_msg="FYI, tensor._base has shape %s, and new_data_tensor has shape %s."
        % ("--" if tensor._base is None else tensor._base.shape, new_data_tensor.shape),
    )
    tensor.data = new_data_tensor
    

class CheckpointingException(Exception):
    """Base checkpointing related exception"""

    pass
    
StateDict = Dict[str, Any]
CommonStateDict = Dict[str, Any]
ShardedStateDict = Dict[str, Any]
ReplicaId = Union[int, Tuple[int, ...]]

class ShardedBase(ABC):
    """Base class for ShardedTensor and ShardedStateDict."""

    key: str
    data: object
    replica_id: ReplicaId

    @abstractmethod
    def validate_metadata_integrity(self):
        """Codifies the constraints on metadata attributes."""

    @abstractmethod
    def without_data(self) -> 'ShardedBase':
        """Returns a new ShardedBase instance with data=None."""
        raise NotImplementedError

@dataclass
class ShardedTensor(ShardedBase):
    """Represents a mapping between a local tensor and a global tensor.

    Global tensor is assumed to consist of many local tensors distributed
    between different processes.

    Args:
        key: unique identifier of a global tensor
        data: local tensor data. Can be None only for consistency validation
        dtype: tensor dtype
        local_shape: local tensor shape
        global_shape: global tensor shape
        global_offset: offset of a local tensor in a global tensor,
            specified in number of tensor elements
        axis_fragmentations: global tensor fragmentation of each axis
        replica_id: indicates given local tensor's replication wrt.
            local tensors in different processes
        prepend_axis_num: number of axes prepended to the local tensor to
            reflect global tensor shape. The behavior is similar to
            unsqueezing the local tensor.
        allow_shape_mismatch: if True, during loading, the global shape of
            a stored tensor does not have to match the expected global shape.
            Useful for representing tensors with flexible shape,
            e.g. padded.
        flattened_range: specifies a slice that should be applied to a
            flattened tensor with `local_shape` in order to get
            the tensor stored as `data`
    """

    key: str
    data: Optional[torch.Tensor] = field(repr=False)
    dtype: torch.dtype
    local_shape: Tuple[int, ...]
    global_shape: Tuple[int, ...]
    global_offset: Tuple[int, ...]
    axis_fragmentations: Optional[Tuple[int, ...]]
    replica_id: ReplicaId = 0
    prepend_axis_num: int = 0
    allow_shape_mismatch: bool = False
    flattened_range: Optional[slice] = None

    def __post_init__(self):
        self.validate_metadata_integrity()

    def validate_metadata_integrity(self) -> None:
        """Codifies the constraints on metadata attributes.

        Meeting those constraints is guaranteed when instantiating a ShardedTensor
        class with `from_rank_offsets` or `from_rank_offsets_flat` constructors.

        Returns:
            None
        """
        has_flattened_range = self.flattened_range is not None
        if self.data is not None:
            if self.data.dtype != self.dtype:
                raise CheckpointingException(
                    f'Data dtype should match `dtype` attribute for {self}'
                )
            if not has_flattened_range and self.data.shape != self.local_shape:
                raise CheckpointingException(
                    f'Data shape should match `local_shape` attribute for {self}'
                )
            if has_flattened_range:
                if self.data.ndim != 1:
                    raise CheckpointingException(f'Data should be 1D for a flattened {self}')
                real_data = self.data
                try:
                    self.data = None
                    self.init_data(device='meta')
                    if self.data.shape != real_data.shape:
                        raise CheckpointingException(
                            f'Data shape {real_data.shape} doesnt match'
                            f' expected {self.data.shape} for {self}'
                        )
                finally:
                    self.data = real_data

        if len(self.global_shape) != len(self.global_offset):
            raise CheckpointingException(
                f'Global offset dimensions should be equal to global shape dimensions for {self}'
            )
        if len(self.local_shape) + self.prepend_axis_num != len(self.global_shape):
            raise CheckpointingException(
                f'Local shape together with `prepend_axis_num` dimensions should be '
                f'equal to global shape dimensions for {self}'
            )

        for off, sh in zip(self.global_offset[self.prepend_axis_num :], self.local_shape):
            # NOTE: In custom FSDP, we have a case where a new parameter shard is created locally.
            # For example, consider parameters [p0, p1, p2] sharded across GPU0 and GPU1.
            # GPU0 receives p0 and a portion of p1, while GPU1 receives the
            # remaining portion of p1 and p2.
            # As a result, there is no parameter shard of p2 on GPU0, and
            # the shape of p2 on GPU0 is zero.
            if sh != 0 and off % sh != 0:
                raise CheckpointingException(
                    f'Global offset ({off}) must be divisible by local shape ({sh}) for {self}.'
                )

        if has_flattened_range and self.flattened_range.step is not None:
            raise CheckpointingException(
                f'`step` argument in the flattened range of a ShardedTensor is not supported.'
            )

    def global_slice(self) -> Tuple[Union[int, slice], ...]:
        """
        Returns a tuple of int and slice objects representing a slice of the
        global tensor that this ShardedTensor corresponds to.
        """
        assert len(self.global_offset) == len(self.local_shape) + self.prepend_axis_num
        return tuple(
            chain(
                (off for off in self.global_offset[: self.prepend_axis_num]),
                (
                    slice(off, off + sh)
                    for off, sh in zip(
                        self.global_offset[self.prepend_axis_num :], self.local_shape
                    )
                ),
            )
        )

    def global_coordinates(self) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of np.ndarrays representing the coordinates of the global tensor
        that this ShardedTensor corresponds to.
        """
        if self.flattened_range is None:
            raise CheckpointingException(
                f'`global_coordinates` is undefined for'
                f' {self.__class__.__name__} without `flattened_range`'
            )

        local_coords = self.local_coordinates()
        assert len(local_coords) + self.prepend_axis_num == len(self.global_offset), (
            len(local_coords),
            self,
        )
        global_coords = tuple(
            c + off
            for c, off in zip((0,) * self.prepend_axis_num + local_coords, self.global_offset)
        )
        return global_coords

    def local_coordinates(self) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of np.ndarrays representing the coordinates of the local tensor
        that this ShardedTensor corresponds to.
        """
        if self.flattened_range is None:
            raise CheckpointingException(
                f'`local_coordinates` is undefined for'
                f' {self.__class__.__name__} without `flattened_range`'
            )

        # TODO: np.unravel_index?
        mask = np.zeros(np.product(self.local_shape), dtype=bool)
        mask[self.flattened_range] = True
        return np.nonzero(mask.reshape(self.local_shape))

    def local_chunk_offset_in_global(self) -> Tuple[int, ...]:
        """Offset of a local chunk in a global array of chunks.

        Returns:
            Tuple[int, ...]: the offset of the whole local chunk in a global array of chunks.
        """
        assert len(self.global_offset) == len(self.local_shape) + self.prepend_axis_num
        chunk_offset = list(self.global_offset[: self.prepend_axis_num])
        for off, sh in zip(self.global_offset[self.prepend_axis_num :], self.local_shape):
            assert off % sh == 0, str(self)
            chunk_offset.append(off // sh)
        return tuple(chunk_offset)

    def max_allowed_chunks(self) -> Tuple[int, ...]:
        """
        Returns the maximum allowed chunks for this ShardedTensor.
        """
        chunks = []
        for axis_sh, axis_fragm in zip(self.global_shape, self.axis_fragmentations):
            if not self.allow_shape_mismatch and axis_sh % axis_fragm != 0:
                raise CheckpointingException(
                    f'Axis shape ({axis_sh}) not divisible by axis fragmentation ({axis_fragm}'
                )
            axis_chunk_size = axis_sh // axis_fragm
            chunks.append(axis_chunk_size)
        return tuple(chunks)

    def without_data(self):
        return replace(self, data=None)

    @classmethod
    def from_rank_offsets(
        cls,
        key: str,
        data: torch.Tensor,
        *rank_offsets: Tuple[int, int, int],
        replica_id: ReplicaId = 0,
        prepend_axis_num: int = 0,
        flattened_range: None = None,
        **init_kwargs,
    ):
        """Allows to construct the ShardedTensor given offset specified in process ranks.

        Args:
            key (str): unique key
            data (torch.Tensor): local tensor data
            rank_offsets (Tuple[int, int, int]): each tuple
                (axis, axis_rank_offset, axis_fragm) says that if
                global tensor is divided into `axis_fragm` fragment along `axis`
                axis, then local tensor data corresponds to the `axis_rank_offset` chunk.
            replica_id (ReplicaId): see ShardedTensor
            prepend_axis_num (int): see ShardedTensor
            flattened_range (None): must be None when using this constructor
            init_kwargs: passed to ShardedTensor.__init__
        """
        if flattened_range is not None:
            raise ValueError(
                'Cannot instantiate a flat ShardedTensor with `from_rank_offsets` method.'
                ' Use `from_rank_offsets_flat` instead'
            )
        global_offset = [0] * (data.ndim + prepend_axis_num)
        global_shape = ([1] * prepend_axis_num) + list(data.shape)
        axis_fragmentations = [1] * (data.ndim + prepend_axis_num)
        _seen_axis = set()
        for axis, axis_rank_offset, axis_fragm in rank_offsets:
            if axis < 0 or axis_rank_offset < 0 or axis_fragm < 1 or axis_rank_offset >= axis_fragm:
                raise CheckpointingException(f'Invalid rank offsets: {rank_offsets} for key {key}.')
            _seen_axis.add(axis)

            local_axis_shape = 1 if axis < prepend_axis_num else data.shape[axis - prepend_axis_num]
            global_shape[axis] = axis_fragm * local_axis_shape
            global_offset[axis] = axis_rank_offset * local_axis_shape
            axis_fragmentations[axis] = axis_fragm

        return cls(
            key,
            data,
            data.dtype,
            tuple(data.shape),
            tuple(global_shape),
            tuple(global_offset),
            tuple(axis_fragmentations),
            replica_id,
            prepend_axis_num,
            flattened_range=flattened_range,
            **init_kwargs,
        )

    @classmethod
    def from_rank_offsets_flat(
        cls,
        key: str,
        data: torch.Tensor,
        non_flat_local_shape: Tuple[int, ...],
        *args,
        flattened_range: Optional[slice] = None,
        **kwargs,
    ):
        """Allows to construct a *flattened* ShardedTensor given offset specified in process ranks.

        Args:
            key (str):
            data (torch.Tensor): this should be a flattened data tensor
            non_flat_local_shape (Tuple[int, ...]): expected local shape of a non-flat chunk
            *args: passed unchanged to the `from_rank_offsets` constructor
            flattened_range (slice): see ShardedTensor. Defaults to None, but must be set to
                a non-None slice.
            **kwargs:

        Returns:
            ShardedTensor: constructed ShardedTensor instance
        """
        if flattened_range is None:
            raise CheckpointingException(
                'Cannot instantiate a non-flat ShardedTensor with `from_rank_offsets_flat` method.'
                ' Use `from_rank_offsets` instead'
            )
        if data.ndim != 1:
            raise CheckpointingException(
                f'Flattened ShardedTensor requires 1D data, got shape: {data.shape}'
            )
        if flattened_range.stop - flattened_range.start != data.numel():
            raise CheckpointingException(
                f'Flattened ShardedTensor data length ({data.numel()}) must meet the '
                f'slice length: {flattened_range.stop - flattened_range.start}'
            )

        non_flat_data_meta = torch.empty(*non_flat_local_shape, dtype=data.dtype, device='meta')
        sh_ten = cls.from_rank_offsets(key, non_flat_data_meta, *args, **kwargs)
        instance = replace(sh_ten, data=data, flattened_range=flattened_range)
        instance.validate_metadata_integrity()
        return instance

    def init_data(self, device: Union[str, torch.device], init_fn=torch.empty):
        """
        Initialize the tensor data of this ShardedTensor.

        Only called if `data` attribute is None.

        Args:
            device (Union[str, torch.device]): device to place the tensor on
            init_fn (Callable, optional): function to use to initialize the tensor.
                Defaults to `torch.empty`.
        """
        if self.data is not None:
            return
        self.data = init_fn(self.local_shape, dtype=self.dtype, device=device)
        if self.flattened_range is not None:
            self.data = self.data.flatten()[self.flattened_range.start : self.flattened_range.stop]

    def narrow(self, dim: int, start: int, length: int) -> List['ShardedTensor']:
        """This is an analogue of torch.narrow for ShardedTensors.

        Narrowing assumes that we narrow a local tensor on each rank.
        This has consequences on local_shape, global_shape, global_offset, etc.

        Args:
            dim (int): dimension to narrow. Doesn't include prepended axes.
            start (int): start element
            length (int): length of the slice

        Returns:
            List[ShardedTensor]: narrowed ShardedTensors. For non-flat tensors,
                the list will always have 1 element. For flat ShardedTensors the number of
                elements varies depending on `dim` and on overlap, because flat
                tensors must be contiguous. In particular the list can be empty.
        """
        prepended_dim = dim + self.prepend_axis_num
        local_length_along_dim = self.local_shape[dim]

        def _update_tuple(x, ind, val):
            x = list(x)
            x[ind] = val
            return tuple(x)

        def _safe_div(x, y):
            assert x % y == 0, (x, y)
            return x // y

        # Decrease global shape and global offset by `length / local_length_along_dim`
        assert (
            self.global_shape[prepended_dim] % local_length_along_dim == 0
        ), f'Only regular grid of local tensors is supported for narrowing, got: {self}'
        assert (
            self.global_offset[prepended_dim] % local_length_along_dim == 0
        ), f'Only regular grid of local tensors is supported for narrowing, got: {self}'
        global_shape = _update_tuple(
            self.global_shape,
            prepended_dim,
            _safe_div(self.global_shape[prepended_dim] * length, local_length_along_dim),
        )
        global_offset = _update_tuple(
            self.global_offset,
            prepended_dim,
            _safe_div(self.global_offset[prepended_dim] * length, local_length_along_dim),
        )

        if self.flattened_range is None:
            new_data = self.data.narrow(dim, start, length)
            # always a single result tensor
            return [
                replace(
                    self,
                    data=new_data,
                    local_shape=new_data.shape,
                    global_shape=global_shape,
                    global_offset=global_offset,
                )
            ]
        else:
            if dim != 0:
                raise CheckpointingException(
                    f'Narrowing along the first axis is supported for now only, got dim={dim}'
                )

            # If dim=0, we will always get 0 or 1 resulting tensor.
            # If dim>1, in general there can be more result tensors (e.g. max 3 for dim=1)

            # For on original flat ShardedTensor of local shape [3, 4] and
            # flattened_range=slice(5, 10),
            # the X signs mark the actual (flat) data in `self.data`
            # notice 12 (3*4) total "virtual" elements, out of which 5 is actual data.
            # flat original: [.....XXXXX..]

            # If we narrow to start=1, length=1 in the original local shape dimensions,
            # the overlapping flat slice would be:
            # narrow to:     [....XXXX....]
            # flat overlap:  [.....XXX....]

            # Now `data` is flattened and sliced, so we must compute local_shape manually
            local_shape = _update_tuple(self.local_shape, dim, length)
            other_dims_volume = np.prod(
                _update_tuple(local_shape, dim, 1)
            )  # 4 in the example above
            volume_before_split = other_dims_volume * start  # 4 in the example above
            volume_of_split = other_dims_volume * length  # 4 in the example above

            flat_slice_start_shifted = (
                self.flattened_range.start - volume_before_split
            )  # 5 - 4 = 1 in the example above
            flat_slice_stop_shifted = (
                self.flattened_range.stop - volume_before_split
            )  # 10 - 4 = 6 in the example above

            # Find an intersection of
            # (flat_slice_start_shifted, flat_slice_stop_shifted) vs (0, volume_of_split)

            if flat_slice_stop_shifted <= 0 or flat_slice_start_shifted >= volume_of_split:
                return []  # no intersection

            # new_flattened_range = slice(1, 4) in the example above
            new_flattened_range = slice(
                max(flat_slice_start_shifted, 0), min(flat_slice_stop_shifted, volume_of_split)
            )
            # Apply the intersection to the flattened data tensor.
            # Compute start and slice appropriate length
            intersection_slice_start = (
                new_flattened_range.start - flat_slice_start_shifted
            )  # 0 in the example above
            new_data = self.data[
                intersection_slice_start : intersection_slice_start
                + new_flattened_range.stop
                - new_flattened_range.start
            ]

            return [
                replace(
                    self,
                    data=new_data,
                    local_shape=local_shape,
                    global_shape=global_shape,
                    global_offset=global_offset,
                    flattened_range=new_flattened_range,
                )
            ]

def prepare_input_tensors_for_wgrad_compute(grad_output, all_gathered_input):
    """Ensure grad_output is stored in a contiguous buffer."""
    # Doing gather + slicing during the NeMo forward pass can make this tensor
    # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
    # clones it if it's not contiguous:
    # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
    grad_output = grad_output.contiguous()
    all_gathered_input = all_gathered_input.contiguous()
    # Convert the tensor shapes to 2D for execution compatibility
    if grad_output.dim() == 3:
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )
        all_gathered_input = all_gathered_input.view(
            all_gathered_input.shape[0] * all_gathered_input.shape[1], all_gathered_input.shape[2]
        )

    return grad_output, all_gathered_input

def get_full_tensor_if_necessary(tensor):
    """For DTensor gets full tensor if some ranks will not have a local copy"""
    need_full_tensor = False
    for i in range(tensor.device_mesh.ndim):
        if (
            isinstance(tensor.placements[i], Shard)
            and tensor.device_mesh.shape[i] > tensor.shape[tensor.placements[i].dim]
        ):
            need_full_tensor = True
            break

    tensor = tensor.full_tensor() if need_full_tensor else tensor._local_tensor

    return tensor
    
def make_sharded_tensor_for_checkpoint(tensor, key, prepend_offsets=(), replica_id=None, **kwargs):
    """Helper for instantiating a non-sharded ShardedTensor (replicated across TP and DP group).

    Optionally, can provide offsets which prepend new dimensions to the tensor.
    """

    prepend_axis_num = len(prepend_offsets)

    new_offsets = []
    dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
    dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=True)
    dp_replica_id = parallel_state.get_data_parallel_rank(with_context_parallel=True)

    if HAVE_DTENSOR and isinstance(tensor, DTensor):
        # FSDP2 sharding
        dp_replica_id = 0
        tensor = get_full_tensor_if_necessary(tensor)
        new_offsets.append((prepend_axis_num, dp_rank, dp_size))

    if replica_id is None:
        replica_id = (0, parallel_state.get_tensor_model_parallel_rank(), dp_replica_id)

    if hasattr(tensor, 'fully_shard_param_local_shard'):
        assert len(replica_id) == 3, f'Expected replica_id format (PP, TP, DP), got: {replica_id}'
        replica_id = (*replica_id[:2], 0)

        sh_ten = ShardedTensor.from_rank_offsets_flat(
            key,
            tensor.fully_shard_param_local_shard,
            tensor.shape,
            *prepend_offsets,
            flattened_range=slice(*tensor.fully_shard_param_local_index),
            replica_id=replica_id,
            prepend_axis_num=prepend_axis_num,
            **kwargs,
        )
        setattr(sh_ten, 'is_data_parallel_fully_shard', True)
        return sh_ten

    return ShardedTensor.from_rank_offsets(
        key,
        tensor,
        *prepend_offsets,
        *new_offsets,
        replica_id=replica_id,
        prepend_axis_num=prepend_axis_num,
        **kwargs,
    )

def make_tp_sharded_tensor_for_checkpoint(
    tensor, key, tp_axis=0, replica_id=None, prepend_offsets=(), **kwargs
):
    """Helper for instantiating a ShardedTensor where the `tp_axis` dimension
    is sharded across TP group.

    Optionally, can provide offsets which prepend new dimensions to the tensor.
    """
    prepend_axis_num = len(prepend_offsets)

    new_offsets = []
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=True)
    dp_replica_id = parallel_state.get_data_parallel_rank(with_context_parallel=True)

    new_offsets.append((tp_axis + prepend_axis_num, tp_rank, tp_size))

    if HAVE_DTENSOR and isinstance(tensor, DTensor):
        # TP + FSDP2 sharding
        dp_replica_id = 0
        tensor = tensor._local_tensor

        if tp_axis == 0:
            # both FSDP2 and TP shards axis 0
            # default MCore uses tp-cp-ep-dp-pp
            # FSDP2 is compatibile with TP, CP
            new_offsets[0] = (prepend_axis_num, tp_rank * dp_size + dp_rank, tp_size * dp_size)
        else:
            # FSDP2 shards axis 0 and TP shards some other axis
            new_offsets.append((prepend_axis_num, dp_rank, dp_size))

    if replica_id is None:
        replica_id = (0, 0, dp_replica_id)

    if hasattr(tensor, 'fully_shard_param_local_shard'):
        assert len(replica_id) == 3, f'Expected replica_id format (PP, TP, DP), got: {replica_id}'
        replica_id = (*replica_id[:2], 0)

        sh_ten = ShardedTensor.from_rank_offsets_flat(
            key,
            tensor.fully_shard_param_local_shard,
            tensor.shape,
            *prepend_offsets,
            (
                tp_axis + prepend_axis_num,
                parallel_state.get_tensor_model_parallel_rank(),
                parallel_state.get_tensor_model_parallel_world_size(),
            ),
            flattened_range=slice(*tensor.fully_shard_param_local_index),
            replica_id=replica_id,
            prepend_axis_num=prepend_axis_num,
            **kwargs,
        )
        setattr(sh_ten, 'is_data_parallel_fully_shard', True)
        return sh_ten

    return ShardedTensor.from_rank_offsets(
        key,
        tensor,
        *prepend_offsets,
        *new_offsets,
        replica_id=replica_id,
        prepend_axis_num=prepend_axis_num,
        **kwargs,
    )

def make_sharded_tensors_for_checkpoint(
    state_dict: StateDict,
    prefix: str,
    tensor_parallel_layers_axis_map: Optional[Dict[str, int]] = None,
    sharded_offsets: Iterable[Tuple[int, int, int]] = (),
    extra_state_suffix: str = '_extra_state',
):
    """Wraps tensors from transformer layers with ShardedTensor or ShardedObject.

    For a given `state_dict`, wraps:
    - all _extra_states with ShardedObject
    - all tensors specified in tensor_parallel_layers_axis_map with TP and DP sharded ShardedTensor
    - other values with DP sharded ShardedTensor

    Args:
        state_dict (StateDict): state_dict to convert
        prefix (str): prefix appended to keys in final state dict
        tensor_parallel_layers_axis_map (Dict[str, int], optional): dict mapping layer
            names to the axis for TP sharding
        sharded_offsets (Iterable[Tuple[int, int, int]], optional): sharding already
            applied (e.g. PP related), passed along to ShardedTensor
        extra_state_suffix (str, default = '_extra_state'): layers with this
            suffix will be wrapped with ShardedObject instead of ShardedTensor.

    """

    if tensor_parallel_layers_axis_map is None:
        tensor_parallel_layers_axis_map = {}

    sharded_state_dict = {}
    for layer_name in state_dict.keys():
        tensor = state_dict[layer_name]
        layer_key = f'{prefix}{layer_name}'

        if layer_name.endswith(extra_state_suffix):
            sharded_state_dict[layer_key] = make_sharded_object_for_checkpoint(
                tensor, layer_key, sharded_offsets
            )

        elif layer_name in tensor_parallel_layers_axis_map:
            tp_axis = tensor_parallel_layers_axis_map[layer_name]
            sharded_state_dict[layer_key] = make_tp_sharded_tensor_for_checkpoint(
                tensor, layer_key, tp_axis, prepend_offsets=sharded_offsets
            )

        else:
            sharded_state_dict[layer_key] = make_sharded_tensor_for_checkpoint(
                tensor, layer_key, prepend_offsets=sharded_offsets
            )

    return sharded_state_dict