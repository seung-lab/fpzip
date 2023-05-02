# cython: language_level=3
from libc.stdio cimport FILE, fopen, fwrite, fclose
from libc.stdlib cimport calloc, free
from libc.stdint cimport uint8_t
from cpython cimport array 
import array
import sys

cimport numpy as numpy

import numpy as np

__VERSION__ = '1.2.1'
__version__ = __VERSION__

FPZ_ERROR_STRINGS = [
  "success",
  "cannot read stream",
  "cannot write stream",
  "not an fpz stream",
  "fpz format version not supported",
  "precision not supported",
  "memory buffer overflow"
]

cdef extern from "fpzip.h":
  ctypedef struct FPZ:
    int type
    int prec
    int nx
    int ny
    int nz
    int nf

  cdef FPZ* fpzip_read_from_file(FILE* file)
  cdef FPZ* fpzip_read_from_buffer(void* buffer) 
  cdef int fpzip_read_header(FPZ* fpz)
  cdef size_t fpzip_read(FPZ* fpz, void* data)
  cdef void fpzip_read_close(FPZ* fpz)
  
  cdef FPZ* fpzip_write_to_file(FILE* file)
  cdef FPZ* fpzip_write_to_buffer(void* buffer, size_t size)
  cdef int fpzip_write_header(FPZ* fpz)
  cdef int fpzip_write(FPZ* fpz, const void* data)
  cdef void fpzip_write_close(FPZ* fpz)

  ctypedef enum fpzipError:
    fpzipSuccess             = 0, # no error 
    fpzipErrorReadStream     = 1, # cannot read stream 
    fpzipErrorWriteStream    = 2, # cannot write stream 
    fpzipErrorBadFormat      = 3, # magic mismatch; not an fpz stream 
    fpzipErrorBadVersion     = 4, # fpz format version not supported 
    fpzipErrorBadPrecision   = 5, # precision not supported 
    fpzipErrorBufferOverflow = 6  # compressed buffer overflow 

  cdef fpzipError fpzip_errno = 0

class FpzipError(Exception):
  pass

class FpzipWriteError(FpzipError):
  pass

class FpzipReadError(FpzipError):
  pass

cpdef allocate(typecode, ct):
  cdef array.array array_template = array.array(chr(typecode), [])
  # create an array with 3 elements with same type as template
  return array.clone(array_template, ct, zero=True)

def validate_order(order):
  order = order.upper()
  if order not in ('C', 'F'):
    raise ValueError("Undefined order parameter '{}'. Options are 'C' or 'F'".format(order))
  return order

def compress(data, precision=0, order='C'):
  """
  fpzip.compress(data, precision=0, order='C')

  Takes up to a 4d numpy array of floats or doubles and returns
  a compressed bytestring.

  precision indicates the number of bits to truncate. Any value above
  zero indicates a lossy operation.

  order is 'C' or 'F' (row major vs column major memory layout) and 
  should correspond to the underlying orientation of the input array.
  """
  if data.dtype not in (np.float32, np.float64):
    raise ValueError("Data type {} must be a floating type.".format(data.dtype))

  order = validate_order(order)

  while len(data.shape) < 4:
    if order == 'C':
      data = data[np.newaxis, ...]
    else: # F
      data = data[..., np.newaxis ]

  if not data.flags['C_CONTIGUOUS'] and not data.flags['F_CONTIGUOUS']:
    data = np.copy(data, order=order)

  header_bytes = 28 # read.cpp:fpzip_read_header + 4 for some reason

  cdef char fptype = b'f' if data.dtype == np.float32 else b'd'
  cdef array.array compression_buf = allocate(fptype, data.size + header_bytes)

  cdef FPZ* fpz_ptr
  if fptype == b'f':
    fpz_ptr = fpzip_write_to_buffer(compression_buf.data.as_floats, data.nbytes + header_bytes)
  else:
    fpz_ptr = fpzip_write_to_buffer(compression_buf.data.as_doubles, data.nbytes + header_bytes)

  if data.dtype == np.float32:
    fpz_ptr[0].type = 0 # float
  else:
    fpz_ptr[0].type = 1 # double

  fpz_ptr[0].prec = precision

  shape = list(data.shape)

  if order == 'C':
    shape.reverse()

  # Dr. Lindstrom noted that fpzip expects czyx order with
  # channels changing most slowly. We should probably change
  # this up in v2 and write in the documentation what should
  # go where.
  fpz_ptr[0].nx = shape[0]
  fpz_ptr[0].ny = shape[1]
  fpz_ptr[0].nz = shape[2]
  fpz_ptr[0].nf = shape[3]

  if fpzip_write_header(fpz_ptr) == 0:
    fpzip_write_close(fpz_ptr)
    del compression_buf
    raise FpzipWriteError("Cannot write header. %s" % FPZ_ERROR_STRINGS[fpzip_errno])

  cdef float[:,:,:,:] arr_memviewf
  cdef double[:,:,:,:] arr_memviewd
  cdef size_t outbytes

  cdef float[:] bufviewf
  cdef double[:] bufviewd

  if data.size == 0:
    fpzip_write_close(fpz_ptr)
    if data.dtype == np.float32:
      bufviewf = compression_buf
      bytes_out = bytearray(bufviewf[:header_bytes])  
    else:
      bufviewd = compression_buf
      bytes_out = bytearray(bufviewd[:header_bytes])  
    del compression_buf
    return bytes(bytes_out)

  if data.dtype == np.float32:
    arr_memviewf = data
    outbytes = fpzip_write(fpz_ptr, <void*>&arr_memviewf[0,0,0,0])
    bufviewf = compression_buf
    bytes_out = bytearray(bufviewf[:outbytes])[:outbytes]
  else:
    arr_memviewd = data
    outbytes = fpzip_write(fpz_ptr, <void*>&arr_memviewd[0,0,0,0])
    bufviewd = compression_buf
    bytes_out = bytearray(bufviewd[:outbytes])[:outbytes]

  del compression_buf
  fpzip_write_close(fpz_ptr)
  
  if outbytes == 0:
    raise FpzipWriteError("Compression failed. %s" % FPZ_ERROR_STRINGS[fpzip_errno])

  return bytes(bytes_out)

def decompress(bytes encoded, order='C'):
  """
  fpzip.decompress(encoded, order='C')

  Accepts an fpzip encoded bytestring (e.g. b'fpy)....') and 
  returns the original array as a 4d numpy array.

  order is 'C' or 'F' (row major vs column major memory layout) and 
  should correspond to the byte order of the originally compressed
  array.
  """
  order = validate_order(order)

  # line below necessary to convert from PyObject to a naked pointer
  cdef unsigned char *encodedptr = <unsigned char*>encoded 
  cdef FPZ* fpz_ptr = fpzip_read_from_buffer(<void*>encodedptr)

  if fpzip_read_header(fpz_ptr) == 0:
    raise FpzipReadError("cannot read header: %s" % FPZ_ERROR_STRINGS[fpzip_errno])

  cdef char fptype = b'f' if fpz_ptr[0].type == 0 else b'd'
  nx, ny, nz, nf = fpz_ptr[0].nx, fpz_ptr[0].ny, fpz_ptr[0].nz, fpz_ptr[0].nf

  cdef array.array buf = allocate(fptype, nx * ny * nz * nf)

  cdef size_t read_bytes = 0;
  if fptype == b'f':
    read_bytes = fpzip_read(fpz_ptr, buf.data.as_floats)
  else:
    read_bytes = fpzip_read(fpz_ptr, buf.data.as_doubles)

  if read_bytes == 0:
    raise FpzipReadError("decompression failed: %s" % FPZ_ERROR_STRINGS[fpzip_errno])

  fpzip_read_close(fpz_ptr)

  dtype = np.float32 if fptype == b'f' else np.float64

  if order == 'C':
    return np.frombuffer(buf, dtype=dtype).reshape( (nf, nz, ny, nx), order='C')
  elif order == 'F':
    return np.frombuffer(buf, dtype=dtype).reshape( (nx, ny, nz, nf), order='F')
  else:
    raise ValueError(f"Undefined order parameter '{order}'. Options are 'C' or 'F'")


