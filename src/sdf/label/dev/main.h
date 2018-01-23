static __global__ void main(Sdf_v sdf_v, int n, const Particle *pp, /**/ int *labels) {
    enum {X, Y, Z};
    int pid;
    Particle p;
    float s;
    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    p = pp[pid];
    s = sdf(&sdf_v, p.r[X], p.r[Y], p.r[Z]);
    labels[pid] =
        s > 2 ? LABEL_DEEP :
        s >=0 ? LABEL_WALL :
                LABEL_BULK;
}
