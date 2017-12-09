__global__ void main(Coords c, float mass, float alpha, int n, const Particle *pp, /**/ Force *ff) {
    enum {X, Y};
    int pid;
    float d, *f;
    const float *r;
    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    r = pp[pid].r;
    f = ff[pid].f;
    d = yl2yc(c, r[Y]); /* coordinate relative to domain center */
    f[X] += mass * alpha * d;
}
