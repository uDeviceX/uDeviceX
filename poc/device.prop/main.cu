#include <stdio.h>

void GetDeviceProperties(struct cudaDeviceProp *prop) {
    cudaError_t e;
    int device;
    device = 0;
    e = cudaGetDeviceProperties (prop, device);
    if (e != cudaSuccess) {
        fprintf(stderr, "GetDeviceProperties failed\n");
        exit(2);
    }
}

int main() {
    cudaDeviceProp p;
    GetDeviceProperties(&p);
    printf("totalGlobalMem = %ld\n", p.totalGlobalMem);
}
