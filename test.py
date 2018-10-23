import pytest

import base64

import numpy as np
import fpzip

def test_doubles():
  x = np.random.random_sample(size=(128, 128, 128, 3)).astype(np.float64)

  x = np.ascontiguousarray(x)
  y = fpzip.compress(x)
  z = fpzip.decompress(y, order='C')

  assert np.all(x == z)

  x = np.asfortranarray(x)
  y = fpzip.compress(x)
  z = fpzip.decompress(y, order='F')

  assert np.all(x == z)

  x = np.random.random_sample(size=(0, 0, 0, 0)).astype(np.float64)
  y = fpzip.compress(x)
  z = fpzip.decompress(y)

  assert np.all(x == z)  

def test_floats():
  x = np.random.random_sample(size=(128, 128, 128, 3)).astype(np.float32)

  x = np.ascontiguousarray(x)
  y = fpzip.compress(x)
  z = fpzip.decompress(y, order='C')

  assert np.all(x == z)

  x = np.asfortranarray(x)
  y = fpzip.compress(x)
  z = fpzip.decompress(y, order='F')

  assert np.all(x == z)

  x = np.random.random_sample(size=(128, 128, 128)).astype(np.float32)

  x = np.ascontiguousarray(x)
  y = fpzip.compress(x)
  z = fpzip.decompress(y)
  z = np.squeeze(z, axis=3)

  assert np.all(x == z)

  x = np.asfortranarray(x)
  y = fpzip.compress(x)
  z = fpzip.decompress(y, order='F')
  z = np.squeeze(z, axis=3)

  assert np.all(x == z)

  x = np.random.random_sample(size=(0, 0, 0, 0)).astype(np.float32)
  
  x = np.ascontiguousarray(x)
  y = fpzip.compress(x)
  z = fpzip.decompress(y)

  assert np.all(x == z)

  x = np.asfortranarray(x)
  y = fpzip.compress(x)
  z = fpzip.decompress(y, order='F')

  assert np.all(x == z)

def test_noncontiguous_memory():
  x = np.random.random_sample(size=(15, 15, 15, 1)).astype(np.float32)
  x_broken = x[2::2,::3,::2]
  
  y = fpzip.compress(x_broken)
  z = fpzip.decompress(y)

  assert np.all(x_broken == z)

def test_basic_conformation():
  # corresponds to the base64 encoded compression of
  # np.array([[[1]]], dtype=np.float32).tobytes()
  # Was compressed by fpzip C program.
  one_fpz = b'ZnB5KYcO8R7gAP8AAAD/AAAA/wAAAP8A8zvT3FAAAAA=\n'
  one_fpz = base64.decodestring(one_fpz) # decodebytes is the modern name, but py27 doesn't have it

  one_array = np.array([[[1]]], dtype=np.float32)
  compressed = fpzip.compress(one_array)

  assert len(one_fpz) == len(compressed)
  
  assert np.all(
    fpzip.decompress(compressed) == fpzip.decompress(one_fpz)
  )

  # encoded np.array([[[1,2,3], [4,5,6]]], dtype=np.float32)
  six_fpz = b'ZnB5KYcO8R7gAP8AAAH+AAAC/QAAAP8A8zvT3GsIJgDU4C0p1pY/+2Z1QAA=\n'
  six_fpz = base64.decodestring(six_fpz)

  six_array = np.array([[[1,2,3], [4,5,6]]], dtype=np.float32)
  compressed = fpzip.compress(six_array)

  assert len(six_fpz) == len(compressed)
  assert np.all(
    fpzip.decompress(six_fpz) == fpzip.decompress(compressed)
  )
