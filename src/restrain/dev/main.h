static __global__ void sum(int color, int n, const Particle *pp, const int *cc) {
    int i, good;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    good = (i < n) && (cc[i] == color);
    sum0(good, pp, i);
}

static __global__ void shift(int color, float3 v, int n, const int *cc, /**/ Particle *pp) {
    int i;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n || cc[i] != color) return;
    shift0(i, v, /**/ pp);
}
