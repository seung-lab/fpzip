[![PyPI version](https://badge.fury.io/py/fpzip.svg)](https://badge.fury.io/py/fpzip)

# fpzip

Python C++ bindings for the fpzip algorithm (version 1.2.0). The version number for this package is independent. 

```python
import fpzip
import numpy as np

data = np.array(..., dtype=np.float32) # 3d or 4d float or double array
compressed_bytes = fpzip.compress(data, precision=0) # b'...'
data_again = fpzip.decompress(compressed_bytes) # 3d or 4d float or double array
```

## Installation

*Python 3 support only. Requires C++ compiler.*

`pip` Installation  

```bash
pip3 install fpzip
```

Direct Installation

```bash
$ pip3 install numpy
$ python3 setup.py develop
```

## References

Algorithm and C++ code by Peter Lindstrom and Martin Isenburg. Cython interface code by William Silversmith. Check out [Dr. Lindstrom's site](https://computation.llnl.gov/projects/floating-point-compression).

1. Peter Lindstrom and Martin Isenburg, “[Fast and Efficient Compression of Floating-Point Data,](https://www.researchgate.net/publication/6715625_Fast_and_Efficient_Compression_of_Floating-Point_Data)” IEEE Transactions on Visualization and Computer Graphics, 12(5):1245-1250, September-October 2006, doi:[10.1109/TVCG.2006.143](http://dx.doi.org/10.1109/TVCG.2006.143).  
