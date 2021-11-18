# Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
# All rights reserved.

import torch
import ps_tensor

# tensor = torch.randn(10, dtype=torch.float, device=torch.device("cuda"))
tensor = torch.randn(2, 3, dtype=torch.half)
print("pytorch tensor", tensor)
print("pytorch data_ptr", tensor.data_ptr())

capsule = ps_tensor.torch_to_ps_tensor(tensor)
print("before move to GPU")
capsule.move_gpu()
print("after move to GPU")
tensor_v2 = ps_tensor.ps_tensor_to_torch(capsule)
print("generate a tensor in C++ and convert it to the torch", tensor_v2)
