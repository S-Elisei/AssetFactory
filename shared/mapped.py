"""Weights held in memory maps of their safetensors files: a model built without weights takes its parameters and
buffers from read-only memory maps, and moves its parameters and buffers to the GPU and back to the maps. Imports torch
and the standard library only."""
import json
import mmap
import struct

import torch

# The dtype of each safetensors header type name that torch.frombuffer reads.
DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64}


def map_tensors(paths):
    """Returns {name: tensor} of the safetensors files `paths`, each tensor backed by a read-only memory map and of the
    dtype named in its header. The tensors keep their maps open."""
    tensors = {}
    for path in paths:
        with open(path, "rb") as file:
            mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        header_size = struct.unpack("<Q", mapped[:8])[0]
        header = json.loads(mapped[8:8 + header_size])
        header.pop("__metadata__", None)
        for key, info in header.items():
            start, end = info["data_offsets"]
            dtype = DTYPES[info["dtype"]]
            tensor = torch.frombuffer(mapped, dtype=dtype, count=(end - start) // dtype.itemsize,
                                      offset=8 + header_size + start)
            tensors[key] = tensor.view(info["shape"])
    return tensors


def attach(module, tensors):
    """Makes the parameters and buffers of `module`, built without weights, the tensors of `tensors` (named as the state
    dict of `module`). Each of them keeps its mapped tensor as `host`."""
    module.load_state_dict(tensors, assign=True)
    for name, tensor in (*module.named_parameters(), *module.named_buffers()):
        if name in tensors:
            tensor.host = tensor.data


def _held(module, recurse):
    return [tensor for tensor in (*module.parameters(recurse=recurse), *module.buffers(recurse=recurse))
            if hasattr(tensor, "host")]


def to_device(module, device, recurse=True):
    """Copies the attached tensors of `module` (of its submodules too when `recurse`) to `device`."""
    for tensor in _held(module, recurse):
        tensor.data = tensor.host.to(device)


def to_host(module, recurse=True):
    """Points the attached tensors of `module` (of its submodules too when `recurse`) back to their memory maps."""
    for tensor in _held(module, recurse):
        tensor.data = tensor.host
