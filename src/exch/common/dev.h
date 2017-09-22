namespace dev {

// TODO this is copy/paste from distr/common
static __device__ void fid2shift(int id, /**/ int s[3]) {
    enum {X, Y, Z};
    s[X] = XS * frag_i2d(id, X);
    s[Y] = YS * frag_i2d(id, Y);
    s[Z] = ZS * frag_i2d(id, Z);
}

static  __device__ void shift_1p(const int s[3], /**/ Particle *p) {
    enum {X, Y, Z};
    p->r[X] += s[X];
    p->r[Y] += s[Y];
    p->r[Z] += s[Z];
}

__global__ void shift_one_frag(int n, const int fid, /**/ Particle *pp) {
    int i, s[3];
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;
    
    fid2shift(fid, /**/ s);
    shift_1p(s, /**/ pp + i);
}

} // dev
