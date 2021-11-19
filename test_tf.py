# Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
# All rights reserved.

import tensorflow as tf
import ps_tensor

# TODO
tensor = tf.random.normal([4], 0, 1, tf.float32)
print(tensor)
capsule = ps_tensor.tf_to_ps_tensor(tensor)
tensorv2 = ps_tensor.ps_tensor_to_tf(capsule)
print(tensorv2)
