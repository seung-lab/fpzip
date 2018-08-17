import os
import setuptools
from setuptools.command.build_ext import build_ext as _build_ext

join = os.path.join
fpzipdir = 'fpzip-1.2.0'

# NOTE: If fpzip.cpp does not exist:
# cython -3 --fast-fail -v --cplus ./ext/src/third_party/fpzip-1.2.0/src/fpzip.pyx

class build_ext(_build_ext):
  def finalize_options(self):
    _build_ext.finalize_options(self)
    # Prevent numpy from thinking it is still in its setup process:
    __builtins__.__NUMPY_SETUP__ = False
    import numpy
    self.include_dirs.append(numpy.get_include())

sources = [ join(fpzipdir, 'src', x) for x in ( 
  'error.cpp', 'rcdecoder.cpp', 'rcencoder.cpp', 
  'rcqsmodel.cpp', 'write.cpp', 'read.cpp', 
) ]

sources += [ 'fpzip.cpp' ]

setuptools.setup(
  setup_requires=[ 'pbr', 'numpy' ],
  install_requires=[ 'numpy' ],
  cmdclass={'build_ext':build_ext},
  extras_require={
    ':python_version == "2.7"': ['futures'],
    ':python_version == "2.6"': ['futures'],
  },
  ext_modules=[
    setuptools.Extension(
      'fpzip',
      sources=sources,
      language='c++',
      include_dirs=[ join(fpzipdir, 'inc') ],
      extra_compile_args=[
        '-std=c++11', 
        '-DFPZIP_FP=FPZIP_FP_FAST', '-DFPZIP_BLOCK_SIZE=0x1000', '-DWITH_UNION',
      ]
    )
  ],
  pbr=True)





