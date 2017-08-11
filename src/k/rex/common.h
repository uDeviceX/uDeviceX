namespace k_rex {
__device__ float fst(float2 s) { return s.x; }
__device__ float scn(float2 s) { return s.y; }

__device__ void i2shift(int fid, /**/ float *d) {
    /* fragment id to coordinate shift */
    enum {X, Y, Z};
    d[X] = - ((fid +     2) % 3 - 1) * XS;
    d[Y] = - ((fid / 3 + 2) % 3 - 1) * YS;
    d[Z] = - ((fid / 9 + 2) % 3 - 1) * ZS;
}
}
