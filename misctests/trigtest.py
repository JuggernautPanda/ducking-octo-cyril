__author__ = 'ashwin'
"""
simple test 2
"""
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

import matplotlib.pyplot as p

mod = SourceModule("""
__global__ void gpusin(float *dest, float *a)
{
  const int i = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));
  dest[i] = sinf(a[i]) ;
}

__global__ void gpucos(float *dest, float *a)
{
  const int i = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));
  dest[i] = cosf(a[i]) ;
}


""")

gpusin = mod.get_function("gpusin")
gpucos = mod.get_function("gpucos")

a = (numpy.arange(4000)).astype(numpy.float32)
b = numpy.sin(a)
c = numpy.cos(a)

destsin = numpy.zeros_like(a)
destcos = numpy.zeros_like(a)

gpusin(drv.Out(destsin), drv.In(a), block=(400, 1, 1), grid=(10, 1, 1))
gpucos(drv.Out(destcos), drv.In(a), block=(400, 1, 1), grid=(10, 1, 1))

p.subplot(2,1,1)
p.plot(destsin)
p.subplot(2,1,2)
p.plot(destcos)
p.show()
