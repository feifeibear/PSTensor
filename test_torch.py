# Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
# All rights reserved.

import torch
import ps_tensor

ps_tensor.set_stderr_verbose_level(0)
# tensor = torch.randn(10, dtype=torch.float, device=torch.device("cuda"))
tensor = torch.randn(2, 3, dtype=torch.float, device=torch.device("cuda"))
print("pytorch tensor", tensor)
print("pytorch tensor data_ptr", tensor.data_ptr())

# hijack memory of PyTorch to our allocator (Here is use cub CachingDeviceAllocator,
# You can implement one by youself.)
flushed_tensor = ps_tensor.ps_flush_payload(tensor)
print("flushed_tensor: ", flushed_tensor)
print("flushed_tensor data_ptr: ", flushed_tensor.data_ptr())

# move tensor to other device.
moved_tensor = ps_tensor.ps_move(flushed_tensor, torch.device('cpu:0'))
print("moved_tensor: ", moved_tensor)
print("moved_tensor data_ptr: ", moved_tensor.data_ptr())

# Test torch.narrow
narrowed_tensor = torch.narrow(moved_tensor.view(-1), 0, 0, 2)
print('narrowed_tensor ', narrowed_tensor)
