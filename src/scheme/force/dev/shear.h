__global__ void main(float mass, float alpha, int n, const Particle *pp, /**/ Force *ff) {
    enum {X, Y};
    int pid;
    float d, *f;
    const float *r;
    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    r = pp[pid].r;
    f = ff[pid].f;
    d = r[Y] - glb::r0[Y]; /* coordinate relative to domain center */
    f[X] += mass * alpha * d;
}
