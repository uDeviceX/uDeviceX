/*
  [t]exture [o]bject

 */

#include <stdio.h>
#include "u.h" /* utils */

/* ceiling `m' to `n' (returns the smallest `A' such n*A is not less
   than `m') */
#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )

/* a common kernel execution configuration */
#define k_cnf(n) ceiln((n), 128), 128

#define n 5 /* number of elements */
float *d_A,   *d_B; /* device */
float  h_A[n], h_B[n]; /* host */
#define sz ((n)*sizeof(h_A[0]))

cudaTextureObject_t   to;
cudaResourceDesc      resD;
cudaTextureDesc       texD;

__global__ void plus(float *A, cudaTextureObject_t to) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;

  float b = tex1Dfetch<float>(to, i);
  if (i < n) A[i] += b;
}

void h_ini() { /* host ini */
  for (int i = 0; i < n; ++i) {
    h_A[i] =    i;
    h_B[i] = 10*i;
  }
}

void d_ini() { /* device ini */
  cudaMalloc(&d_A, sz);
  cudaMalloc(&d_B, sz);
}

void h2d() { /* host to device */
  cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);
}

void d2h() { /* device to host */
  cudaMemcpy(h_A, d_A, sz, cudaMemcpyDeviceToHost);
}

void tex_ini() {
  memset(&resD, 0, sizeof(resD));
  resD.resType = cudaResourceTypeLinear;
  resD.res.linear.devPtr  = d_B;
  resD.res.linear.sizeInBytes = sz;
  resD.res.linear.desc = cudaCreateChannelDesc<float>();

  memset(&texD, 0, sizeof(texD));
  texD.normalizedCoords = 0;
  texD.readMode = cudaReadModeElementType;

  checkCudaErrors(cudaCreateTextureObject(&to, &resD, &texD, NULL));
}

int main() {
  h_ini(); d_ini();

  h2d(); /* host to device */
  tex_ini();

  plus<<<k_cnf(n)>>>(d_A, to);
  d2h(); /* device to host */

  for (int i = 0; i < n; i++)
    printf("a[%d] = %2g\n", i, h_A[i]);
}
