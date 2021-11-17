# Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
# All rights reserved.

import torch
import torch.utils.dlpack as dlpack
import ps_tensor.ps_tensor_cxx as cxx

# import turbo_transformers.turbo_transformers_cxx as cxx

# tensor = torch.randn(10, dtype=torch.float, device=torch.device("cuda"))
tensor = torch.randn(2, 3, dtype=torch.half)
print("pytorch tensor", tensor)
print("pytorch data_ptr", tensor.data_ptr())


def convert2ps_tensor(t):
    return cxx.Tensor.from_dlpack(dlpack.to_dlpack(t))


ps_tensor = convert2ps_tensor(tensor)
prank_res = ps_tensor.prank()
print(prank_res)
th = dlpack.from_dlpack(prank_res.to_dlpack())
print("generate a tensor in C++ and convert it to the torch", th)


print("ps_tensor ndim", ps_tensor.n_dim())
print("ps_tensor float data", ps_tensor.float_data())
# print("show ps_tensor")
# ps_tensor.print()
print("release ps_tensor")
del ps_tensor
print("release finished")

ps_tensor = convert2ps_tensor(th)
ps_tensor.print()
print("show ps_tensor again")
