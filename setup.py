import os
import setuptools
from setuptools import Command
import pkg_resources
import sys

from setuptools.command.build_ext import build_ext as _build_ext

# Use the numpy installation strategy pandas uses
class build_ext(_build_ext):
  def finalize_options(self):
    numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

    for ext in self.extensions:
      if (hasattr(ext, 'include_dirs') and \
        numpy_incl not in ext.include_dirs):

        ext.include_dirs.append(numpy_incl)

    _build_ext.build_extensions(self)

class DummyBuildSrc(Command):
    """ numpy's build_src command interferes with Cython's build_ext.
    """
    user_options = []

    def initialize_options(self):
        self.py_modules_dict = {}

    def finalize_options(self):
        pass

    def run(self):
        pass


join = os.path.join
fpzipdir = 'fpzip-1.2.0'

# NOTE: If fpzip.cpp does not exist:
# cython -3 --fast-fail -v --cplus ./ext/src/third_party/fpzip-1.2.0/src/fpzip.pyx


sources = [ join(fpzipdir, 'src', x) for x in (
  'error.cpp', 'rcdecoder.cpp', 'rcencoder.cpp',
  'rcqsmodel.cpp', 'write.cpp', 'read.cpp',
) ]

sources += [ 'fpzip.cpp' ]

setuptools.setup(
  cmdclass={
    'build_src': DummyBuildSrc,
    'build_ext': build_ext,
  },
  setup_requires=['pbr', 'numpy'],
  install_requires=['numpy'],
  extras_require={
    ':python_version == "2.7"': ['futures'],
    ':python_version == "2.6"': ['futures'],
  },
  ext_modules=[
    setuptools.Extension(
      'fpzip',
      sources=sources,
      language='c++',
      include_dirs=[ join(fpzipdir, 'inc')],
      extra_compile_args=[
        '-std=c++11',
        '-DFPZIP_FP=FPZIP_FP_FAST', '-DFPZIP_BLOCK_SIZE=0x1000', '-DWITH_UNION',
      ]
    )
  ],
  pbr=True)
