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

// brak cudaDeviceSynchronize skutkuje ryzykiem odczytania przez cpu danych wciąż jeszcze nie zsynchronizowanych z gpu. 
// Tak jakbyśmy wywołali print(h_x[n]) przed  
//cudaMemcpy(h_x,d_x,nsize*sizeof(float),
  //              cudaMemcpyDeviceToHost)
// 

__device__ __managed__ float x[16+1];
__global__ void my_first_kernel()
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = (float) threadIdx.x+blockIdx.x;
}
//
// main code
//

int main(int argc, const char **argv)
{
 // float *x;
  int   nblocks, nthreads, nsize, n; 

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  //checkCudaErrors(cudaMallocManaged(&x, nsize*sizeof(float)));

  // execute kernel
  
  my_first_kernel<<<nblocks,nthreads>>>();
  getLastCudaError("my_first_kernel execution failed\n");

  // synchronize to wait for kernel to finish, and data copied back
  
  cudaDeviceSynchronize();

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,x[n]);

  // free memory 

  //checkCudaErrors(cudaFree(x));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
