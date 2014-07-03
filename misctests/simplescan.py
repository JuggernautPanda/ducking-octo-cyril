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

const int a =  threadIdx.x ;


if(a<(1000-1) && a>1)
{

dest[a] = (2*src[a]+src[a+1]+src[a-1])/3;
}

else
{

dest[a]=0;

}

}
"""
    )




src = np.sin(np.arange(0,100,0.1)).astype(np.float32)
dest = np.zeros(src.shape).astype(np.float32)

scan = mod.get_function("scan")
scan(drv.Out(dest),  drv.In(src), block=(1024,1,1), grid=(1,1,1))

p.subplot(2,1,1)
p.plot(src)
p.subplot(2,1,2)
p.plot(dest)
p.show()