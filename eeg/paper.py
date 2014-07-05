__author__ = 'ashwin'

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule
import coeff as c

mod = SourceModule("""
#include<stdio.h>
__global__ void filter(float *res,float *coeff, float *input, int *n)
{
  const int tid  = threadIdx.x+(blockIdx.x*(blockDim.x));
  int i=0;
  float sum=0;
  int N = n[0];

  if(tid<(65536-N))
  {
  sum=0;
  for(i=0;i<N;i++)
  {
  sum+=coeff[i]*input[tid+i];
  }
  res[tid] = sum;
  }

  else
  {
  res[tid] = 0;
  }

}
""")

filter = mod.get_function("filter")

input = np.random.randn(65536).astype(np.float32)
res = (input)

###################################################################
# alpha
###################################################################

coeff = np.array(c.alpha).astype(np.float32)
Ntap = np.int32([len(coeff)]).astype(np.int32)

print Ntap[0]

filter(
        drv.Out(res), drv.In(coeff), drv.In(input),drv.In(Ntap),
        block=(1024,1,1),grid=(64,1,1))

print res

###################################################################
# beta
###################################################################

coeff = np.array(c.beta).astype(np.float32)
Ntap = np.int32([len(coeff)]).astype(np.int32)

print Ntap[0]

filter(
        drv.Out(res), drv.In(coeff), drv.In(input),drv.In(Ntap),
        block=(1024,1,1),grid=(64,1,1))

print res

###################################################################
# theta
###################################################################

coeff = np.array(c.theta).astype(np.float32)
Ntap = np.int32([len(coeff)]).astype(np.int32)

print Ntap[0]

filter(
        drv.Out(res), drv.In(coeff), drv.In(input),drv.In(Ntap),
        block=(1024,1,1),grid=(64,1,1))

print res

###################################################################
# delta
###################################################################

coeff = np.array(c.delta).astype(np.float32)
Ntap = np.int32([len(coeff)]).astype(np.int32)

print Ntap[0]

filter(
        drv.Out(res), drv.In(coeff), drv.In(input),drv.In(Ntap),
        block=(1024,1,1),grid=(64,1,1))

print res