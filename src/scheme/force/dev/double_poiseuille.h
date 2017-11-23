__global__ void main(float mass, Param par, int n, const Particle *pp, /**/ Force *ff) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    const float *r = pp[pid].r;
    float       *f = ff[pid].f;

    float d = r[Y] - glb::r0[Y]; /* coordinate relative to domain
                                     center */
    if (d <= 0) par.a *= -1;
    f[X] += mass * par.a;
}
