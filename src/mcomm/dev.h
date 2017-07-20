__global__ void shift(const float3 s, const int n, /**/ Particle *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    Particle p = pp[i];
    p.r[0] += s.x; p.r[1] += s.y; p.r[2] += s.z;
    pp[i] = p;
}
