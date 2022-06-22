from .fastsimrust import *

import numpy as np 

def _as_numpy_array(self):
    return np.array(list(self))

setattr(Pyo3ArrayU32, "__array__", _as_numpy_array)
setattr(Pyo3ArrayF64, "__array__", _as_numpy_array)
setattr(Pyo3ArrayBool, "__array__", _as_numpy_array)
setattr(Pyo3VecF64, "__array__", _as_numpy_array)