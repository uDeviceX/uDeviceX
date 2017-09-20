#include <stdio.h>

#define N 10
int *a;

__global__ void uninit(int *a) {
    print
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
