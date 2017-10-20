__global__ void particle2float4(const Particle *src, const int n, float4 *dst) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    Particle p = src[pid];
    dst[pid] = make_float4(p.r[X], p.r[Y], p.r[Z], 0);
}

__global__ void float42particle(const float4 *src, const int n, Particle *dst) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    const float4 r = src[pid];
    Particle p;
    p.r[X] = r.x; p.r[Y] = r.y; p.r[Z] = r.z;
    p.v[X] = p.v[Y] = p.v[Z] = 0;
    dst[pid] = p;
}
