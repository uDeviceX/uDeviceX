namespace dev {
static __device__ int box(const float r[3]) {
    /* which neighboring point belongs to? */
    enum {X, Y, Z};
    int c;
    int vc[3]; /* vcode */
    int   L[3] = {XS, YS, ZS};
    for (c = 0; c < 3; ++c)
        vc[c] = (2 + (r[c] >= -L[c]/2) + (r[c] >= L[c]/2)) % 3;
    return vc[X] + 3 * (vc[Y] + 3 * vc[Z]);
}

static __global__ void halo(const Particle *pp, const int n, /**/ int *iidx[], int size[]) {
    int pid, code, entry;
    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    const Particle *p = &pp[pid];
    code = box(p->r);
    if (code > 0) {
        entry = atomicAdd(size + code, 1);
        iidx[code][entry] = pid;
    }
}

} /* namespace */
