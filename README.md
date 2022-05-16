[![PyPI version](https://badge.fury.io/py/fpzip.svg)](https://badge.fury.io/py/fpzip)

# fpzip

fpzip is a compression algorithm supporting lossless and lossy encoding for up to 4 dimensional floating point data. This package contains Python C++ bindings for the fpzip algorithm (version 1.3.0). The version number for this package is independent. Python 3.7+ is supported.

```python
import fpzip
import numpy as np

data = np.array(..., dtype=np.float32) # up to 4d float or double array
# Compress data losslessly, interpreting the underlying buffer in C (default) or F order.
compressed_bytes = fpzip.compress(data, precision=0, order='C') # returns byte string
# Back to 3d or 4d float or double array, decode as C (default) or F order.
data_again = fpzip.decompress(compressed_bytes, order='C') 
```

## Installation

#### `pip` Binary Installation  

```bash
pip install fpzip
```

If we have a precompiled binary available the above command should just work. However, if you have to compile from sounce, it's unfortunately necessary to install numpy first because of a quirk in the Python installation procedure that won't easily recognize when a numpy installation completes in the same process. There are some hacks, but I haven't gotten them to work.

#### `pip` Source Installation

*Requires C++ compiler.*

```bash
pip install numpy
pip install fpzip
```

#### Direct Installation

*Requires C++ compiler.*  

```bash
$ pip install numpy
$ python setup.py develop
```

## References

Algorithm and C++ code by Peter Lindstrom and Martin Isenburg. Cython interface code by William Silversmith. Check out [Dr. Lindstrom's site](https://computing.llnl.gov/projects/fpzip) or the [fpzip Github page](https://github.com/LLNL/fpzip).

1. Peter Lindstrom and Martin Isenburg, "[Fast and Efficient Compression of Floating-Point Data,](https://www.researchgate.net/publication/6715625_Fast_and_Efficient_Compression_of_Floating-Point_Data)" IEEE Transactions on Visualization and Computer Graphics, 12(5):1245-1250, September-October 2006, doi:[10.1109/TVCG.2006.143](http://dx.doi.org/10.1109/TVCG.2006.143).  
