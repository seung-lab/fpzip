import pytest

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
