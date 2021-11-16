import torch
import torch.utils.dlpack as dlpack
import ps_tensor.ps_tensor_cxx as cxx


tensor = torch.randn(10)
print('pytorch tensor', tensor)
print('pytorch data_ptr', tensor.data_ptr())

# torch -> ps_tensor
def convert2ps_tensor(t):
    return cxx.Tensor.from_dlpack(dlpack.to_dlpack(t))

ps_tensor = convert2ps_tensor(tensor)
print(ps_tensor.n_dim())
print(ps_tensor.float_data())
print("show ps_tensor")
ps_tensor.print()

# ps_tensor -> torch
torch_v2 = dlpack.from_dlpack(ps_tensor.to_dlpack())
print(torch_v2)
# print(ps_tensor.n_dim())

# torch move to gpu
tensor = tensor.cuda()
print("show ps_tensor again")
ps_tensor.print()