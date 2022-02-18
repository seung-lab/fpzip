import os
import setuptools
import sys

import numpy as np

join = os.path.join
fpzipdir = 'fpzip-1.3.0'

# NOTE: If fpzip.cpp does not exist:
# cython -3 --fast-fail -v --cplus ./ext/src/third_party/fpzip-1.2.0/src/fpzip.pyx

sources = [ 
  join(fpzipdir, 'src', x) for x in ( 
    'error.cpp', 'rcdecoder.cpp', 'rcencoder.cpp', 
    'rcqsmodel.cpp', 'write.cpp', 'read.cpp', 
  ) 
]
sources += [ 'fpzip.cpp' ]

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
  setup_requires=['pbr', 'numpy'],
  python_requires="~=3.6", # >= 3.6 < 4.0
  ext_modules=[
    setuptools.Extension(
      'fpzip',
      sources=sources,
      language='c++',
      include_dirs=[ join(fpzipdir, 'include'), np.get_include() ],
      extra_compile_args=extra_compile_args,
    )
  ],
  pbr=True)





