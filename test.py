import torch
import torch.utils.dlpack as dlpack
import ps_tensor.ps_tensor_cxx as cxx


tensor = torch.randn(10)

def convert2tt_tensor(t):
    return cxx.Tensor.from_dlpack(dlpack.to_dlpack(t))

b = convert2tt_tensor(tensor)
print(b.n_dim())
print(b.float_data())