try:
    # `ps_tensor_cxxd` is the name on debug mode
    import ps_tensor.ps_tensor_cxxd as cxx
except ImportError:
    import ps_tensor.ps_tensor_cxx as cxx
import contextlib
