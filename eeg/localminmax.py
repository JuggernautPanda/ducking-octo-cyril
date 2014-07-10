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
__global__ void localmax( float *o, float *i)
{

const int t =  threadIdx.x ;
if(t>1 && t<1000)
    {
    if((i[t] >= i[t+1]) && (i[t] >= i[t-1]) )
        {
            o[t] = i[t];
        }
    else
        {
        o[t] = 0;
        }
    }
}

__global__ void localmin( float *o, float *i)
{

const int t =  threadIdx.x ;
if(t>1 && t<1000)
    {
    if((i[t] <= i[t+1]) && (i[t] <= i[t-1]) )
        {
            o[t] = i[t];
        }
    else
        {
        o[t] = 0;
        }
    }
}
"""
    )



input= np.random.randn(1000).astype(np.float32)
output = np.zeros(input.shape).astype(np.float32)
output2 = np.zeros(input.shape).astype(np.float32)

localmax = mod.get_function("localmax")
localmax(drv.Out(output),  drv.In(input), block=(1000,1,1), grid=(1,1,1))
localmin = mod.get_function("localmin")
localmin(drv.Out(output2),  drv.In(input), block=(1000,1,1), grid=(1,1,1))


p.subplot(3,1,1)
p.plot(input)
p.subplot(3,1,2)
p.stem(output)
p.subplot(3,1,3)
p.stem(output2)

p.show()