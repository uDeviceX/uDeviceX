__global__ void ini_rnd(long seed, int n, curandState_t *rr) {
    long i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, i, 0, &rr[i]);
}
