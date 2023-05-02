import os
import setuptools
import sys

import numpy as np

class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__

join = os.path.join
fpzipdir = 'fpzip-1.3.0'

sources = [ 
  join(fpzipdir, 'src', x) for x in ( 
    'error.cpp', 'rcdecoder.cpp', 'rcencoder.cpp', 
    'rcqsmodel.cpp', 'write.cpp', 'read.cpp', 
  ) 
]
sources += [ 'fpzip.pyx' ]

extra_compile_args = [
  '-DFPZIP_FP=FPZIP_FP_FAST', 
  '-DFPZIP_BLOCK_SIZE=0x1000', 
  '-DWITH_UNION',
]

if sys.platform == 'darwin':
  extra_compile_args += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]

if sys.platform == 'win32':
  extra_compile_args += [ '/std:c++11', '/O2' ]
else:
  extra_compile_args += [ '-O3', '-std=c++11' ]

setuptools.setup(
  setup_requires=['pbr', 'numpy','cython'],
  python_requires=">=3.7,<4.0", # >= 3.6 < 4.0
  ext_modules=[
    setuptools.Extension(
      'fpzip',
      sources=sources,
      language='c++',
      include_dirs=[ join(fpzipdir, 'include'), NumpyImport() ],
      extra_compile_args=extra_compile_args,
    )
  ],
  pbr=True)





