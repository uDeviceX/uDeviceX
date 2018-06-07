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

__global__ void pp2f4_pos(int n, const float *pp, /**/ float4 *zpp) {
    enum {X, Y, Z};
    int i;
    const float *r, *v;
    float x, y, z;
    float vx, vy, vz;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    r = &pp[6*i];
    v = &pp[6*i + 3];

    x =  r[X];  y = r[Y];  z = r[Z];
    vx = v[X]; vy = v[Y]; vz = v[Z];

    zpp[2*i]     = make_float4(x,   y,  z, 0);
    zpp[2*i + 1] = make_float4(vx, vy, vz, 0);
}
