static __global__ void sum(Map m, int n, const Particle *pp) {
    int i, good;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    good = (i < n) && goodp(m, i);
    sum0(good, pp, i); /* collective */
}

static __global__ void shift(Map m, float3 v, int n, /**/ Particle *pp) {
    int i, good;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    good = (i < n) && goodp(m, i);
    if (good) shift0(i, v, /**/ pp);
}
