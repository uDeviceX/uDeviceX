__global__ void ini_rnd(long seed, int n, curandState_t *rr) {
    long i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &rr[i]);
}

__global__ void ini_flux(int n, /**/ curandState_t *rr, float *cumflux) {
    long i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    cumflux[i] = curand_uniform(rr + i);
}
