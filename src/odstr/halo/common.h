namespace dev {
static __device__ int box(const float r[3]) {
    /* which neighboring point belongs to? */
    enum {X, Y, Z};
    int c;
    int vc[3]; /* vcode */
    int   L[3] = {XS, YS, ZS};
    check(r);
    for (c = 0; c < 3; ++c)
        vc[c] = (2 + (r[c] >= -L[c]/2) + (r[c] >= L[c]/2)) % 3;
    return vc[X] + 3 * (vc[Y] + 3 * vc[Z]);
}

/* [reg]ister a particle */
static __device__ void reg(int pid, int fid, /**/ int *iidx[], int size[]) {
    int entry;
    if (fid > 0) {
        entry = atomicAdd(size + fid, 1);
        iidx[fid][entry] = pid;
    }
}

static __global__ void halo(const Particle *pp, const int n, /**/ int *iidx[], int size[]) {
    int pid;
    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    const Particle p = pp[pid];
    reg(pid, box(p.r), iidx, size);
}

} /* namespace */
