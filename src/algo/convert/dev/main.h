__global__ void pp2rr_current (long n, const Particle *pp, Positioncp *rr) {
    enum {X, Y, Z};
    const float *s;
    float *d;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    d = rr[i].rc;
    s = pp[i].r;

    d[X] = s[X];
    d[Y] = s[Y];
    d[Z] = s[Z];
}

__global__ void pp2rr_previous(long n, const Particle *pp, Positioncp *rr) {
    enum {X, Y, Z};
    const float *s;
    float *d;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    d = rr[i].rp;
    s = pp[i].r;

    d[X] = s[X];
    d[Y] = s[Y];
    d[Z] = s[Z];
}
