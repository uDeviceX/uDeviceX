#include <stdio.h>

#define N 1
int *a;

__global__ void uninit(int *a) {
    printf("a[0]: %d\n", a[0]);
}


void run_uninit() {
    uninit<<<1,1>>>(a);
    cudaGetErrorString(cudaGetLastError());
    printf("Sync: %s\n", cudaGetErrorString(cudaThreadSynchronize()));
}

int main() {
    cudaMalloc(&a, N*sizeof(a[0]));

    run_uninit();

    cudaDeviceReset();
    cudaFree(a);
    return 0;
}
