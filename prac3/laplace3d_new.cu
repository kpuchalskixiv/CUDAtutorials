// Kacper Puchalski


// zad3. X=256, Y=2
// zad4. X=64, Y=1, Z=8

// zad5. zła wersja nvprof - do aktualizacji i odpalenia w domu

// zad6. Padding poprawił wydajność o ~1%
// TODO czytanie i zapisytwanioe w pętli zad7. 3.398 GB/s , 100x mniej niż device bandwidth geforce3060 - 360GB/s





//
// Program to solve Laplace equation on a regular 3D grid
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// define kernel block size
////////////////////////////////////////////////////////////////////////

#define BLOCK_X 64
#define BLOCK_Y 1
#define BLOCK_Z 8

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

#include <laplace3d_kernel_new.h>

////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////

void Gold_laplace3d(long long NX, long long NY, long long NZ,
		    float* h_u1, float* h_u2);

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

  int NX=512, NY=512, NZ=512,
            REPEAT=200;
  long long bx, by, bz, i, j, k, ind;
  float    *h_u1, *h_u2, *h_foo,
           *d_u1, *d_u2, *d_foo;
  
  size_t    bytes = sizeof(float) * NX*NY*NZ;

  printf("Grid dimensions: %d x %d x %d \n\n", NX, NY, NZ);

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory for arrays

  h_u1 = (float *)malloc(bytes);
  h_u2 = (float *)malloc(bytes);
  checkCudaErrors( cudaMalloc((void **)&d_u1, bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_u2, bytes) );

  // initialise u1

  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {
        ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1)
          h_u1[ind] = 1.0f;           // Dirichlet b.c.'s
        else
          h_u1[ind] = 0.0f;
      }
    }
  }

  // copy u1 to device

  cudaEventRecord(start);
  checkCudaErrors( cudaMemcpy(d_u1, h_u1, bytes,
                              cudaMemcpyHostToDevice) );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("Copy u1 to device: %.1f (ms) \n\n", milli);

  // Gold treatment

  cudaEventRecord(start);
  /*
  for (i=0; i<REPEAT; i++) {
    Gold_laplace3d(NX, NY, NZ, h_u1, h_u2);
    h_foo = h_u1; h_u1 = h_u2; h_u2 = h_foo;   // swap h_u1 and h_u2
  }
  */

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("%dx Gold_laplace3d: %.1f (ms) \n\n", REPEAT, milli);
  
  // Set up the execution configuration

  bx = 1 + (NX-1)/BLOCK_X;
  by = 1 + (NY-1)/BLOCK_Y;
  bz = 1 + (NZ-1)/BLOCK_Z;

  dim3 dimGrid(bx,by,bz);
  dim3 dimBlock(BLOCK_X,BLOCK_Y,BLOCK_Z);

  // Execute GPU kernel

  cudaEventRecord(start);

  for (i=0; i<REPEAT; i++) {
    GPU_laplace3d<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_u1, d_u2);
    getLastCudaError("GPU_laplace3d execution failed\n");

    d_foo = d_u1; d_u1 = d_u2; d_u2 = d_foo;   // swap d_u1 and d_u2
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("%dx GPU_laplace3d_new: %.1f (ms) \n\n", REPEAT, milli);

  // Read back GPU results

  cudaEventRecord(start);
  checkCudaErrors( cudaMemcpy(h_u2, d_u1, bytes, cudaMemcpyDeviceToHost) );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("Copy u2 to host: %.1f (ms) \n\n", milli);

  // error check

  float err = 0.0;

  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {
        ind = i + j*NX + k*NX*NY;
        err += (h_u1[ind]-h_u2[ind])*(h_u1[ind]-h_u2[ind]);
      }
    }
  }

  printf("rms error = %f \n",sqrt(err/ (float)(NX*NY*NZ)));
    
 // Release GPU and CPU memory

  checkCudaErrors( cudaFree(d_u1) );
  checkCudaErrors( cudaFree(d_u2) );
  free(h_u1);
  free(h_u2);

  cudaDeviceReset();
}
