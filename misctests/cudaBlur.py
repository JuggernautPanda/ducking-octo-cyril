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
#define INDEX(a, b) a*256+b

__global__ void blur(float *dest ,float *r_img)
{



const int a =  threadIdx.x + blockIdx.x * blockDim.x;
const int b =  threadIdx.y + blockIdx.y * blockDim.y;
float temp = 0;
int i=0;


float c[4]={0,0,0,0};

if ((a>1 && a<255) && (b>1 && b<255))
{
temp = 0;
for(i=0;i<5;i++)
{
  c[0] = r_img[INDEX(a, b)];

temp += c[i];
}
}
__syncthreads();

dest[INDEX(a, b)]=temp;

}

"""
    )

a = scm.imread('Lenna.png').astype(np.float32)

r_img = a[:, :, 0].reshape(65536, order='F')
g_img = a[:, :, 1].reshape(65536, order='F')
b_img = a[:, :, 2].reshape(65536, order='F')


dest_r=r_img
print dest_r.shape


blur = mod.get_function("blur")
blur(drv.Out(dest_r),  drv.In(r_img), block=(8, 8, 1), grid=(32,32, 1))


a[:, :, 0] = np.reshape(dest_r, (256, 256), order='F')
#a[:, :, 1] = np.reshape(dest_b, (256, 256), order='F')
#a[:, :, 2] = np.reshape(dest_g, (256, 256), order='F')


p.imshow(a)
p.show()