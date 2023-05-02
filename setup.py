import os
import setuptools
import sys

import numpy as np

join = os.path.join
fpzipdir = 'fpzip-1.3.0'

# NOTE: If fpzip.cpp does not exist:
# cython --fast-fail -v --cplus fpzip.pyx

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

prerelease_version = os.getenv("PRERELEASE_VERSION", "")
if prerelease_version == "release":
    prerelease_version = ""

setuptools.setup(
    version='1.2.1' + prerelease_version,
    setup_requires=['pbr', 'numpy'],
    python_requires="~=3.6", # >= 3.6 < 4.0
    version = '1.2.1',
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
