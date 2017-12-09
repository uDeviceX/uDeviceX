__global__ void main(Coords c, float mass, float f0, int n, const Particle *pp, /**/ Force *ff) {
    enum {X, Y};
    int pid;
    float *f;
    float d[2], S;
    const float *r;

    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    r = pp[pid].r;
    f = ff[pid].f;
     /* coordinate relative to domain center */
    d[X] = xl2xc(c, r[X]);
    d[Y] = yl2yc(c, r[Y]);

    S = 1.f / (d[X] * d[X] + d[Y] * d[Y]);

    f[X] += mass * S * f0 * d[X];
    f[Y] += mass * S * f0 * d[Y];
}
