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

#include<stdio.h>

__global__ void blur(float *dest ,float *img)
{

int a,b;

int tid =  threadIdx.x + blockIdx.x * blockDim.x;

a=tid/256;
b=tid%256;

if((( a>1) && (a<255)) && ((b>1) && (b<255)))
{

dest[a*256+b] = img[a*256+b];//+img[(a-1)*256+(b-1)]+img[(a+1)*256+(b+1)]+img[(a-1)*256+(b+1)]+img[(a+1)*256+(b-1)]);


}
}
"""
    )

a = scm.imread('Lenna.png').astype(np.float32)
b = a

r_img = a[:, :, 0].reshape(65536, order='F')
g_img = a[:, :, 1].reshape(65536, order='F')
b_img = a[:, :, 2].reshape(65536, order='F')


dest_r = r_img
dest_g = g_img
dest_b = b_img

print dest_r.shape


blur = mod.get_function("blur")
blur(drv.Out(dest_r),  drv.In(r_img), block=(1024,1, 1), grid=(64,1, 1))
#blur(drv.Out(dest_g),  drv.In(g_img), block=(8, 8, 1), grid=(32,32, 1))
#blur(drv.Out(dest_b),  drv.In(b_img), block=(8, 8, 1), grid=(32,32, 1))


b[:, :, 0] = np.reshape(dest_r, (256, 256), order='F')
#b[:, :, 1] = np.reshape(dest_g, (256, 256), order='F')
#b[:, :, 2] = np.reshape(dest_b, (256, 256), order='F')

print a[:,:,0]
print b[:,:,0]

p.subplot(1,2,1)
p.imshow(-a)
p.subplot(1,2,2)
p.imshow(b[:,:,0])

p.show()