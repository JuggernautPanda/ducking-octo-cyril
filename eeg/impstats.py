__author__ = 'ashwin'


import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
import numpy.linalg as la
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x+(blockIdx.x*(blockDim.x));

}
""")

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(65536).astype(numpy.float32)
b = numpy.random.randn(65536).astype(numpy.float32)

dest = numpy.zeros_like(a)
multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(1024,1,1),grid=(64,1,1))

print dest-a*b
