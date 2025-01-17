//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


//
// kernel routine
// 

__global__ void my_first_kernel(float *x, float c)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = c+blockIdx.x;
}

__global__ void addition_kernel(float *x, float *y, float *z)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  z[tid] = y[tid]+x[tid];
}

//
// main code
//

int main(int argc, const char **argv)
{
  float *x, *y, *z, *h_z;
  int   nblocks, nthreads, nsize, n; 

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array
  h_z = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc(&x, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc(&y, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc(&z, nsize*sizeof(float)));

  // execute kernel
  
  my_first_kernel<<<nblocks,nthreads>>>(x, 1);
  my_first_kernel<<<nblocks,nthreads>>>(y, 2);
  addition_kernel<<<nblocks,nthreads>>>(x,y,z);
  getLastCudaError("my_first_kernel execution failed\n");
 
  checkCudaErrors( cudaMemcpy(h_z,z,nsize*sizeof(float),
                 cudaMemcpyDeviceToHost) );
  // synchronize to wait for kernel to finish, and data copied back


	

  printf("vector Z:");
  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_z[n]);

  // free memory 

  checkCudaErrors(cudaFree(x));
  free(h_z);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
