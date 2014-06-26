__author__ = 'ashwin'

#import pycuda.driver as drv
#import pycuda.tools
#import pycuda.autoinit
import numpy
#from pycuda.compiler import SourceModule
import numpy as np
import scipy.misc as scm

lena=scm.imread('Lenna.png').astype(np.uint8)
print lena.shape

#ok