__global__ void filter(float3 origin, int n, const Particle *pp, Params params, /**/ int *kk) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    Particle p = pp[i];
    kk[i] = predicate(origin, params, p.r);
}

