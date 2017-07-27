namespace k_rex {
__global__ void ini() { g::failed = false; }

__device__ void fid2dr(int fid, /**/ float *d) {
    /* fragment id to coordinate shift */
    enum {X, Y, Z};
    d[X] = - ((fid +     2) % 3 - 1) * XS;
    d[Y] = - ((fid / 3 + 2) % 3 - 1) * YS;
    d[Z] = - ((fid / 9 + 2) % 3 - 1) * ZS;
}
}
