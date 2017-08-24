namespace dev {
__global__ void halo(const Particle *pp, const int n, /**/ int *iidx[], int size[]) {
    int pid, code, entry;
    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    const Particle *p = &pp[pid];
    code = k_common::box(p->r);
    if (code > 0) {
        entry = atomicAdd(size + code, 1);
        iidx[code][entry] = pid;
    }
}

} /* namespace */
