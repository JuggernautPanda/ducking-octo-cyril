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




#define INDEX(a, b) a*256+b

__global__ void blur(float *dest,float *r_img)
{



const uint a =  threadIdx.x + blockIdx.x * blockDim.x;
const uint b =  threadIdx.y + blockIdx.y * blockDim.y;

dest[INDEX(a, b)] = r_img[INDEX(a, b)] + r_img[INDEX(a-1, b+1)];

__syncthreads();
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

p.gray()
p.imshow(a[:,:,0])
p.show()