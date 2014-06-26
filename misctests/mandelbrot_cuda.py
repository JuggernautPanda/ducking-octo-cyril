####################################################
#  Calculate Mandelbrot set and save it as a bmp image
#
#  Data parallel version using Pycuda
#  Create string with cuda code and let the graphics
#  card farm out the work to each warp.
#
####################################################
 
import bmp
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as nm
 
# maximal number of iterations to compute a pixel
MAX_ITER = 256
 
# image dimensions
from sizes import nx,ny
 
from pycuda.elementwise import ElementwiseKernel
complex_gpu = ElementwiseKernel(
        "int nx, int ny, int maxiter, int *iteration",
        """
        float zr, zi, z2;
        float jf = 1.0f*(i%ny);
        float rowf = 1.0f*(i/ny);
        float nxf = 1.0f*nx;
        float nyf = 1.0f*ny;
        float qif = 4.0f*rowf/nyf-2.0f;
        float qrf = 4.0f*jf/nxf-2.0f;
        iteration[i] = maxiter;
        zr = 0.0f;
        zi = 0.0f;
        for(int n=0;n < maxiter;n++) {
            float nzr = zr*zr - zi*zi + qrf;
            float nzi =   2.0*zr*zi   + qif;
            zi = nzi;
            zr = nzr;
            z2 = zr*zr+zi*zi;
            if(z2 > 4.0f) {
                iteration[i] = n;
                break;
            }
        }
        """,
        "mandlebrot_gpu",)
 
# allocate a numpy array
iterations = nm.zeros(nx*ny).astype(nm.int32)
 
# allocate a gpu array
iterations_gpu = gpuarray.to_gpu(iterations)
 
# perform the gpu calculation
complex_gpu(nm.int16(nx), nm.int16(ny), nm.int16(MAX_ITER), iterations_gpu)
 
# copy data from the gpu array to the numpy array
iterations_gpu.get(iterations)
 
# reshape the array to look the way we want
image = iterations.reshape(nx,ny)
 
bmp.write_image('image.bmp', nx, ny, image, MAX_ITER)