namespace sdstr {
static __global__ void shiftpp_dev(const int n, const float3 s, /**/ Particle *pp) {
    enum {X, Y, Z};
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float *r = pp[i].r;
        r[X] += s.x; r[Y] += s.y; r[Z] += s.z;
    }
}
}
