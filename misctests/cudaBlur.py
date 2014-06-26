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

__global__ void rgb2gray(int *dest,int *r_img, int *g_img, int *b_img)
{

unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a = idx/256;
  unsigned int b = idx%256;

if((a > 1 && a<255) && (b > 1 && b<255))
{
r_img[INDEX(a, b)] = (0.2(r_img[INDEX(a-1, b-1)]+r_img[INDEX(a+1, b-1)]+r_img[INDEX(a-1, b+1)]+r_img[INDEX(a+1, b+1)]+r_img[INDEX(a, b)]))
g_img[INDEX(a, b)] = (0.2(g_img[INDEX(a-1, b-1)]+g_img[INDEX(a+1, b-1)]+g_img[INDEX(a-1, b+1)]+r_img[INDEX(a+1, b+1)]+g_img[INDEX(a, b)]))
b_img[INDEX(a, b)] = (0.2(b_img[INDEX(a-1, b-1)]+b_img[INDEX(a+1, b-1)]+b_img[INDEX(a-1, b+1)]+b_img[INDEX(a+1, b+1)]+b_img[INDEX(a, b)]))

}
__syncthreads();

}

"""
    )

a = scm.imread('Lenna.png').astype(np.uint8)
r_img = a[:, :, 0].reshape(65536, order='F')
g_img = a[:, :, 1].reshape(65536, order='F')
b_img = a[:, :, 2].reshape(65536, order='F')
dest=r_img
rgb2gray = mod.get_function("rgb2gray")
rgb2gray(drv.In(r_img), drv.In(g_img),drv.In(b_img),block=(1024, 1, 1), grid=(64, 1, 1))

r_img=mod.get()
b_img=mod.get()
g_img=mod.get()

r_img=np.reshape(r_img,(256,256), order='F')
g_img=np.reshape(g_img,(256,256), order='F')
b_img=np.reshape(b_img,(256,256), order='F')



p.gray()
p.imshow(dest)
p.show()