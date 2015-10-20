
import numpy as np


#loop-up lookup for 16 bit
N = 16; lookup = np.asarray([bin(i).count('1') for i in range(1<<N)])


# bit count with uint32 packed bit
def count_bits_u32(u32):
  u32 = (u32 & 0x55555555) + ((u32 & 0xAAAAAAAA) >> 1)
  u32 = (u32 & 0x33333333) + ((u32 & 0xCCCCCCCC) >> 2)
  u32 = (u32 & 0x0F0F0F0F) + ((u32 & 0xF0F0F0F0) >> 4)
  u32 = (u32 & 0x00FF00FF) + ((u32 & 0xFF00FF00) >> 8)
  u32 = (u32 & 0x0000FFFF) + ((u32 & 0xFFFF0000) >> 16)
  return u32


def bit_count_parallel_32(n):
  """
  Return the number of set bits (1) present in the integer number 'n'.
  This algorithm accepts only 32-bit non-negative integer numbers.
  """
  #assert 0 <= n < 2**32, ('Argument of bit_count_parallel_32() must be ' 'non-negative and less than %d (2**32)') % 2**32
  n = n - ((n >> 1) & 0x55555555)
  n = (n & 0x33333333) + ((n >> 2) & 0x33333333)
  return (((n + (n >> 4) & 0x0F0F0F0F) * 0x01010101) & 0xffffffff) >> (8 + 16)


def bit_count_parallel_64(n):
  """
  Return the number of set bits (1) present in the integer number 'n'.
  This algorithm accepts only 64-bit non-negative integer numbers.
  """
  #assert 0 <= n < 2**64, ('Argument of bit_count_parallel_64() must be ' 'non-negative and less than %d (2**64)') % 2**64
  n = n - ((n >> 1) & 0x5555555555555555)
  n = (n & 0x3333333333333333) + ((n >> 2) & 0x3333333333333333)
  return (((n + (n >> 4) & 0x0F0F0F0F0F0F0F0F) * 0x0101010101010101) & 0xffffffffffffffff) >> (8 + 16 + 32)
