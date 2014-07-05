__author__ = 'ashwin'

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
import coeff as c
import matplotlib.pyplot as p
import scipy.io as sc


mod = SourceModule("""

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

mat=sc.loadmat('sample.mat')
input= ((mat['val'][0])[0:65536]).astype(np.float32)


#input = np.random.randn(65536).astype(np.float32)


###################################################################
# alpha
###################################################################
res = np.zeros(input.shape)
coeff = np.array(c.alpha).astype(np.float32)
Ntap = np.int32([len(coeff)]).astype(np.int32)

print Ntap[0]

filter(
        drv.Out(res), drv.In(coeff), drv.In(input),drv.In(Ntap),
        block=(1024,1,1),grid=(64,1,1))


alpha_out = res
print res

###################################################################
# beta
###################################################################
res = np.zeros(input.shape)
coeff = np.array(c.beta).astype(np.float32)
Ntap = np.int32([len(coeff)]).astype(np.int32)

print Ntap[0]

filter(
        drv.Out(res), drv.In(coeff), drv.In(input),drv.In(Ntap),
        block=(1024,1,1),grid=(64,1,1))

beta_out = res

###################################################################
# theta
###################################################################
res = np.zeros(input.shape)
coeff = np.array(c.theta).astype(np.float32)
Ntap = np.int32([len(coeff)]).astype(np.int32)

print Ntap[0]

filter(
        drv.Out(res), drv.In(coeff), drv.In(input),drv.In(Ntap),
        block=(1024,1,1),grid=(64,1,1))

theta_out = res

###################################################################
# delta
###################################################################
res = np.zeros(input.shape)
coeff = np.array(c.delta).astype(np.float32)
Ntap = np.int32([len(coeff)]).astype(np.int32)

print Ntap[0]

filter(
        drv.Out(res), drv.In(coeff), drv.In(input),drv.In(Ntap),
        block=(1024,1,1),grid=(64,1,1))

delta_out = res

##################################################################
# plotting for the sake of clarity
##################################################################

p.subplot(5,1,1)
p.plot(input)
p.title("input")
p.subplot(5,1,2)
p.plot(alpha_out)
p.title("alpha")
p.subplot(5,1,3)
p.plot(beta_out)
p.title("beta")
p.subplot(5,1,4)
p.plot(theta_out)
p.title("theta")
p.subplot(5,1,5)
p.plot(delta_out)
p.title("delta")
p.show()
