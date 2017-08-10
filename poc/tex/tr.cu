/*
  [t]exture [r]eference example

 */

#include <stdio.h>

/* a common texture setup */
#define setup_texture(T, TYPE) do {		     \
    (T).channelDesc = cudaCreateChannelDesc<TYPE>(); \
    (T).filterMode = cudaFilterModePoint;	     \
    (T).mipmapFilterMode = cudaFilterModePoint;	     \
    (T).normalized = 0;				     \
} while (false)

/* ceiling `m' to `n' (returns the smallest `A' such n*A is not less
   than `m') */
#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )

/* a common kernel execution configuration */
#define k_cnf(n) ceiln((n), 128), 128

#define n 5 /* number of elements */
float *d_A,   *d_B; /* device */
float  h_A[n], h_B[n]; /* host */
#define sz ((n)*sizeof(h_A[0]))

texture<float, cudaTextureType1D> tex;

__global__ void plus(float *A) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    float b = tex1Dfetch(tex, i);
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

void h2d() {
  cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);
}

void d2h() {
  cudaMemcpy(h_A, d_A, sz, cudaMemcpyDeviceToHost);
}

void bind() {
  size_t offset;
  setup_texture(tex,  float);
  cudaBindTexture(&offset, tex, d_B, tex.channelDesc, sz);
}

int main() {
  h_ini(); d_ini();

  h2d();
  bind();
  
  plus<<<k_cnf(n)>>>(d_A);
  d2h();

  for (int i = 0; i < n; i++)
    printf("a[%d] = %2g\n", i, h_A[i]);
}
