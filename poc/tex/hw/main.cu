/*
  hello world : add two vectors

 */

#include <stdio.h>

/* ceiling `m' to `n' (returns the smallest `A' such n*A is not less
   than `m') */
#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )

/* a common kernel execution configuration */
#define k_cnf(n) ceiln((n), 128), 128

#define n 5 /* number of elements */
float h_A[n], h_B[n], h_C[n]; /* host */
float *d_A, *d_B, *d_C; /* device */
#define sz ((n)*sizeof(h_A[0]))

__global__ void add(float *A, float *B, float *C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void h_ini() {
  int i;
  for (i = 0; i < n; ++i) {
    h_A[i] =    i;
    h_B[i] = 10*i;
  }
}

void d_ini() {
  cudaMalloc(&d_A, sz);
  cudaMalloc(&d_B, sz);
  cudaMalloc(&d_C, sz);
}

void h2d() {
  cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);
}

void d2h() {
  cudaMemcpy(h_C, d_C, sz, cudaMemcpyDeviceToHost);
}

int main() {
  h_ini();
  d_ini();

  h2d();
  add<<<k_cnf(n)>>>(d_A, d_B, d_C);
  d2h();

  int i;
  for (i = 0; i < n; i++)
    printf("c[%d] = %2g\n", i, h_C[i]);
}
