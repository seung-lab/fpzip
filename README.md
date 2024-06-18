[![GitHub release; latest by date](https://img.shields.io/github/v/release/SETI/rms-fpzip)](https://github.com/SETI/rms-fpzip/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/SETI/rms-fpzip)](https://github.com/SETI/rms-fpzip/releases)
[![Test Status](https://img.shields.io/github/actions/workflow/status/SETI/rms-fpzip/run-tests.yml?branch=main)](https://github.com/SETI/rms-fpzip/actions)
[![Code coverage](https://img.shields.io/codecov/c/github/SETI/rms-fpzip/main?logo=codecov)](https://codecov.io/gh/SETI/rms-fpzip)
<br />
[![PyPI - Version](https://img.shields.io/pypi/v/rms-fpzip)](https://pypi.org/project/rms-fpzip)
[![PyPI - Format](https://img.shields.io/pypi/format/rms-fpzip)](https://pypi.org/project/rms-fpzip)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rms-fpzip)](https://pypi.org/project/rms-fpzip)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rms-fpzip)](https://pypi.org/project/rms-fpzip)
<br />
[![GitHub commits since latest release](https://img.shields.io/github/commits-since/SETI/rms-fpzip/latest)](https://github.com/SETI/rms-fpzip/commits/main/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/SETI/rms-fpzip)](https://github.com/SETI/rms-fpzip/commits/main/)
[![GitHub last commit](https://img.shields.io/github/last-commit/SETI/rms-fpzip)](https://github.com/SETI/rms-fpzip/commits/main/)
<br />
[![Number of GitHub open issues](https://img.shields.io/github/issues-raw/SETI/rms-fpzip)](https://github.com/SETI/rms-fpzip/issues)
[![Number of GitHub closed issues](https://img.shields.io/github/issues-closed-raw/SETI/rms-fpzip)](https://github.com/SETI/rms-fpzip/issues)
[![Number of GitHub open pull requests](https://img.shields.io/github/issues-pr-raw/SETI/rms-fpzip)](https://github.com/SETI/rms-fpzip/pulls)
[![Number of GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed-raw/SETI/rms-fpzip)](https://github.com/SETI/rms-fpzip/pulls)
<br />
![GitHub License](https://img.shields.io/github/license/SETI/rms-fpzip)
[![Number of GitHub stars](https://img.shields.io/github/stars/SETI/rms-fpzip)](https://github.com/SETI/rms-fpzip/stargazers)
![GitHub forks](https://img.shields.io/github/forks/SETI/rms-fpzip)

# Introduction

This is a fork of https://github.com/seung-lab/fpzip with changes to allow it to work
with Python 3.11 and 3.12; it also has a different test and PyPI deployment system.
We are grateful to William Silversmith for all of the hard work necessary to make this
project in the first place.

This fork is maintained by the Ring-Moon Systems Node of NASA's Planetary Data System.

# fpzip

fpzip is a compression algorithm supporting lossless and lossy encoding for up to 4 dimensional floating point data. This package contains Python C++ bindings for the fpzip algorithm (version 1.3.0). The version number for this package is independent. Python 3.9+ is supported. This
package works with both NumPy 1.x and 2.x.

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
pip install rms-fpzip
```

If we have a precompiled binary available the above command should just work. However, if you have to compile from source, it's unfortunately necessary to install numpy first because of a quirk in the Python installation procedure that won't easily recognize when a numpy installation completes in the same process. There are some hacks, but I haven't gotten them to work.

#### `pip` Source Installation

*Requires C++ compiler.*

```bash
pip install numpy
pip install rms-fpzip
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
