## PSTensor : Custimized a Tensor Data Structure Compatible with PyTorch and TensorFlow.


You may need this software in the following cases.
1. Manage memory allocation by yourself. Sometimes, you are irritated by the framework's memory allocation mechanism. They use a complicated caching-based allocator and generate fragments.

2. Unified framework-agnostic memory management operations. For example, you are developing a plugin for both PyTorch and TensorFlow. It spits the tensors to multiple GPUs before an operator and merges them afterward. You have to write different Python logic for PyTorch and TensorFlow, respectively, and can not make sure they work in the same efficiency. Alternatively, you can write C++/CUDA code for these operators and provide two sets of Python APIs for both TF and Torch.

3. Customized Communication Pattern. Using PyTorch, it is impossible to implement GPU P2P communication, since nccl backend only supports collective communication APIs. Now, you can implement it with help of CUDA-level libraries.


## Installation
```
mkdir build && cd build && cmake .. && make
pip install `find . -name "*whl"`
```


## Usage
See [PyTorch Example](./test_torch.py) and [TensorFlow Example](./test_tf.py)  for details.
More features are Working In Progress.
