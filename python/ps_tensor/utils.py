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
