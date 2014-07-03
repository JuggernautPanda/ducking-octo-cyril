__author__ = 'ashwin'

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as p

mod = SourceModule \
    (
"""

__global__ void scan(float *dest,float *src)
{

const int a =  threadIdx.x + blockIdx.x * blockDim.x;

__shared__ float tp[32];
int i=0;
int sum =0;

if(a<(3200-32))
{
for(i=0;i<32;i++)
{
tp[i]= src[a];

}
__syncthreads();
}

if(a<(3200-32) )
{
sum=0;
for(i=0;i<32;i++)
{

sum += (tp[i]);

}

dest[a]=sum;

__syncthreads();
}

else
{

dest[a]=0;
__syncthreads();
}

}
"""
    )




src = a = np.arange(3200).astype(np.float32)
dest = np.zeros(src.shape).astype(np.float32)

scan = mod.get_function("scan")
scan(drv.Out(dest),  drv.In(src), block=(32,1,1), grid=(100,1,1))

p.subplot(2,1,1)
p.plot(src)
p.subplot(2,1,2)
p.plot(dest)
p.show()