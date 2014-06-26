__author__ = 'ashwin'

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule
import numpy as np
import scipy.misc as scm


mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));
  dest[i] = a[i] * b[i];
}
""")

lena=scm.imread('Lenna.png').astype(np.uint8)
print lena.shape

