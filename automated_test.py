import pytest

import base64

import numpy as np
import fpzip

def test_doubles():
  size = (128, 127, 126, 2)
  for dims in range(4):
    print(dims)
    x = np.random.random_sample(size=size[:dims]).astype(np.float64)

    x = np.ascontiguousarray(x)
    y = fpzip.compress(x, order='C')
    z = fpzip.decompress(y, order='C')

    assert np.all(x == z)

    x = np.asfortranarray(x)
    y = fpzip.compress(x, order='F')
    z = fpzip.decompress(y, order='F')
    z = np.squeeze(z)

    assert np.all(x == z)

    x = np.random.random_sample(size=[0] * dims).astype(np.float64)
    y = fpzip.compress(x)
    z = fpzip.decompress(y)

    assert np.all(x == z)  

def test_floats():
  size = (128, 127, 126, 2)
  for dims in range(4):
    print(dims)
    x = np.random.random_sample(size=size[:dims]).astype(np.float64)

    x = np.ascontiguousarray(x)
    y = fpzip.compress(x)
    z = fpzip.decompress(y, order='C')

    assert np.all(x == z)

    x = np.asfortranarray(x)
    y = fpzip.compress(x, order='F')
    z = fpzip.decompress(y, order='F')
    z = np.squeeze(z)

    assert np.all(x == z)

  x = np.random.random_sample(size=(128, 128, 128)).astype(np.float32)

  x = np.ascontiguousarray(x)
  y = fpzip.compress(x)
  z = fpzip.decompress(y)
  z = np.squeeze(z, axis=0) # is3d logic

  assert np.all(x == z)

  x = np.asfortranarray(x)
  y = fpzip.compress(x, order='F')
  z = fpzip.decompress(y, order='F')
  z = np.squeeze(z, axis=3)

  assert np.all(x == z)

  x = np.random.random_sample(size=(0, 0, 0, 0)).astype(np.float32)
  
  x = np.ascontiguousarray(x)
  y = fpzip.compress(x, order='C')
  z = fpzip.decompress(y, order='C')

  assert np.all(x == z)

  x = np.asfortranarray(x)
  y = fpzip.compress(x, order='F')
  z = fpzip.decompress(y, order='F')

  assert np.all(x == z)

def test_noncontiguous_memory():
  x = np.random.random_sample(size=(15, 15, 15, 1)).astype(np.float32)
  x_broken = x[2::2,::3,::2]
  
  for order in ('C', 'F'):
    y = fpzip.compress(x_broken, order=order)
    z = fpzip.decompress(y, order=order)

  assert np.all(x_broken == z)

def test_basic_conformation():
  # corresponds to the base64 encoded compression of
  # np.array([[[1]]], dtype=np.float32).tobytes()
  # Was compressed by fpzip C program.
  one_fpz = b'ZnB5KYcO8R7gAP8AAAD/AAAA/wAAAP8A8zvT3FAAAAA=\n'
  one_fpz = base64.decodebytes(one_fpz) # decodebytes is the modern name, but py27 doesn't have it

  one_array = np.array([[[1]]], dtype=np.float32)
  compressed = fpzip.compress(one_array)

  assert len(one_fpz) == len(compressed)
  
  assert np.all(
    fpzip.decompress(compressed) == fpzip.decompress(one_fpz)
  )

  # encoded np.array([[[1,2,3], [4,5,6]]], dtype=np.float32)
  # with open("six.raw", 'wb') as f:
  #   f.write(six_array.tobytes('C'))
  #
  # ./fpzip -i six.raw -3 3 2 1 -o six.fpz
  #    >> outbytes=44 ratio=0.55
  #
  #  with open('six.fpz', 'rb') as f:
  #    encoded = f.read()
  #  six_fpz = base64.encodestring(encoded)   
  
  six_fpz = b'ZnB5KYcO8R7gAv0AAAH+AAAA/wAAAP8A8zvT3GsIJgDRE0yNUZgAHeZbgAA=\n' # 3 2 1 
  six_fpz = base64.decodebytes(six_fpz)

  six_array = np.array([[[1,2,3], [4,5,6]]], dtype=np.float32)
  compressed = fpzip.compress(six_array)

  assert len(six_fpz) == len(compressed)
  assert np.all(
    fpzip.decompress(six_fpz) == fpzip.decompress(compressed)[0,:,:,:]
  )

def test_oversize_compression():
  # This array compresses to a larger size than the input array
  # This test ensures that compressing this array does not lead to buffer
  # overflows
  arr = np.array([1e-12, 0, 1e-12])
  compressed = fpzip.compress(arr)
  assert np.all(
    fpzip.decompress(compressed).flatten() == arr
  )

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_buffer_overflow(dtype):
  # This data compresses to be larger than the input
  x = np.array([
    0.1376666,  0.1400608,  0.1328782,  0.1352724,  0.1340753,  0.130484,
    0.1340753,  0.1352724,  0.1400608,  0.12808979, 0.12689269, 0.12210429,
    0.12928689, 0.1448492,  0.1412579,  0.12449849, 0.1316811,  0.1316811,
    0.130484,   0.1388637,  0.130484,   0.1388637,  0.130484,   0.1448492,
    0.12808979, 0.12689269, 0.1400608,  0.12569559, 0.11611878, 0.1328782,
    0.12928689, 0.1340753,  0.12090719, 0.12569559, 0.12210429, 0.12569559,
    0.12330139, 0.11611878, 0.1340753,  0.12330139, 0.1352724,  0.130484,
    0.1376666,  0.12449849, 0.12449849, 0.12569559, 0.12090719, 0.12569559,
    0.12330139, 0.12689269, 0.12689269
  ], dtype=dtype)

  x[1:] = np.diff(x)  # transform to array of deltas based off first item (I think, I'm crap with numpy)
  y = fpzip.compress(x)
  print(len(y))
  print(x.nbytes)

