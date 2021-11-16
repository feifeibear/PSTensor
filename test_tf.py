import tensorflow as tf
import ps_tensor.ps_tensor_cxx as cxx

# torch -> ps_tensor
def convert2ps_tensor(t):
    return cxx.Tensor.from_dlpack(tf.experimental.dlpack.to_dlpack(t))


tf_tensor = tf.zeros([2,3])
print(tf_tensor)
dl_tensor = convert2ps_tensor(tf_tensor)
print(dl_tensor.n_dim())