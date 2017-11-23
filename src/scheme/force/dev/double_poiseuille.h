__global__ void main(float mass, float f0, int n, const Particle *pp, /**/ Force *ff) {
    enum {X, Y};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    const float *r = pp[pid].r;
    float       *f = ff[pid].f;

    float d = r[Y] - glb::r0[Y]; /* coordinate relative to domain
                                     center */
    if (d <= 0) f0 *= -1;
    f[X] += mass * f0;
}
