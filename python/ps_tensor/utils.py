# Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
# All rights reserved.

import torch
import torch.utils.dlpack as dlpack
import ps_tensor.ps_tensor_cxx as cxx

set_stderr_verbose_level = cxx.set_stderr_verbose_level


def torch_to_ps_tensor(t) -> cxx.Tensor:
    return cxx.Tensor.from_dlpack(dlpack.to_dlpack(t))


def ps_tensor_to_torch(t) -> torch.Tensor:
    return dlpack.from_dlpack(t.to_dlpack())


def ps_flush_payload(t: torch.Tensor) -> torch.Tensor:
    """
    Copy the payload to a memmory allocated by ourself.
    Delete the memory allocated by PyTorch allocator.
    """
    capsule = torch_to_ps_tensor(t)
    capsule_clone = capsule.clone()
    del t
    del capsule
    return ps_tensor_to_torch(capsule_clone)


def ps_move(t: torch.Tensor, target_device: torch.device) -> torch.Tensor:
    """
    Move a torch tensor to target device.
    1. transfer torch tensor to capsule.
    2. use move func of the capsule to copy payload to a new capsule.
    """
    if t.device == target_device:
        return t
    capsule = torch_to_ps_tensor(t)
    capsule_clone = capsule.move(target_device.type)
    del capsule
    del t
    return ps_tensor_to_torch(capsule_clone)
