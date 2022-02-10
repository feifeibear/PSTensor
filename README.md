## PSTensor : Customized a Tensor Data Structure Compatible with PyTorch and TensorFlow.


You may need this software in the following cases.
1. Manage memory allocation by yourself. Sometimes, you are irritated by the framework's memory allocation mechanism. They use a complicated caching-based allocator and generate fragments.

2. Unified framework-agnostic memory management operations.

3. Customized Communication Pattern. Using PyTorch, it is impossible to implement GPU P2P communication, since nccl backend only supports collective communication APIs. Now, you can implement it with help of CUDA-level libraries.


## Installation
```
mkdir build && cd build && cmake .. && make
pip install `find . -name "*whl"`
```


## Usage
See [PyTorch Example](./test_torch.py) and [TensorFlow Example](./test_tf.py)  for details.
More features are Working In Progress.
