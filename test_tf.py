# Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
# All rights reserved.

import tensorflow as tf
import ps_tensor.ps_tensor_cxx as cxx

# import turbo_transformers.turbo_transformers_cxx as cxx
# torch -> ps_tensor


def convert2ps_tensor(t):
    return cxx.Tensor.from_dlpack(tf.experimental.dlpack.to_dlpack(t))


tf_tensor = tf.zeros([2, 3], dtype=tf.half)
print(tf_tensor)
dl_tensor = convert2ps_tensor(tf_tensor)
print(dl_tensor.n_dim())
dl_tensor.print()
