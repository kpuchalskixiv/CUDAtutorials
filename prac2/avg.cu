
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ float a,b,c;

__constant__ int samples, N;
////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////


__global__ void average_val_poly2(float *d_z, float *d_res)
{
  float res=0.0f, z=0.0f;

  int ind;
  ind = threadIdx.x + blockIdx.x*blockDim.x;
  //ind = threadIdx.x + N*blockIdx.x*blockDim.x;

  for(int i=0; i<samples; i++){
    z=d_z[ind+i];
    res+=a*z*z+b*z+c;
  }
  res/=samples;
  d_res[ind]=res;

}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
  
  int     NBLOCKS=128, NTHREADS=32;
  int h_samples=200;
  float   h_a, h_b, h_c;
  float  *d_z, *d_res, *h_res;
  double  sum1;

  h_a     = 1.0f;
  h_b     = 2.0f;
  h_c     = 5.0f;


  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device
  h_res = (float *)malloc(sizeof(float)*NBLOCKS*NTHREADS);


  checkCudaErrors( cudaMalloc((void **)&d_res, sizeof(float)*NBLOCKS*NTHREADS) );
  checkCudaErrors( cudaMalloc((void **)&d_z, h_samples*sizeof(float)*NTHREADS*NBLOCKS) );

  // define constants and transfer to GPU

  checkCudaErrors( cudaMemcpyToSymbol(a,   &h_a,   sizeof(h_a)) );
  checkCudaErrors( cudaMemcpyToSymbol(b, &h_b, sizeof(h_b)) );
  checkCudaErrors( cudaMemcpyToSymbol(c, &h_c, sizeof(h_c)) );
  checkCudaErrors( cudaMemcpyToSymbol(samples, &h_samples, sizeof(h_samples)) );
  checkCudaErrors( cudaMemcpyToSymbol(samples, &h_samples, sizeof(h_samples)) );

  // random number generation

  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );

  //cudaEventRecord(start);
  checkCudaErrors( curandGenerateNormal(gen, d_z, h_samples*NTHREADS*NBLOCKS, 0.0f, 1.0f) );
  //cudaEventRecord(stop);

  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&milli, start, stop);

  // execute kernel and time it

  cudaEventRecord(start);

  average_val_poly2<<<NBLOCKS, NTHREADS>>>(d_z, d_res);
  getLastCudaError("average_val_poly2 execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("Poly2 kernel execution time (ms): %f \n",milli);

  // copy back results

  checkCudaErrors( cudaMemcpy(h_res, d_res, sizeof(float)*NBLOCKS*NTHREADS,
                   cudaMemcpyDeviceToHost) );

  // compute average

  sum1 = 0.0;
  for (int i=0; i<NBLOCKS+NTHREADS; i++) {
    sum1 += h_res[i];
  }
  sum1/=NBLOCKS+NTHREADS;

  printf("\nAverage value and standard deviation of error  = %13.8f \n\n",
	 sum1);

  // Tidy up library

  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  free(h_res);
  checkCudaErrors( cudaFree(d_res) );
  checkCudaErrors( cudaFree(d_z) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

}
