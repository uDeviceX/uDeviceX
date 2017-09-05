__device__ float2 get(const float2 *pp) { return *pp; }
__device__ Pa pp2p(const float2 *pp, int i) {
    /* pp array to a particle */
    Pa p;
    pp += 3*i;
    p.s0 = get(pp++); p.s1 = get(pp++); p.s2 = get(pp++);
    return p;
}

__device__ void p2pp(Pa p, int i, /**/ float2 *pp) {
    i *= 3;
    pp[i++] = p.s0;
    pp[i++] = p.s1;
    pp[i++] = p.s2;
}

__device__ void p2xyz(const Pa p, /**/ float *x, float *y, float *z) {
    *x = fst(p.s0);
    *y = scn(p.s0);
    *z = fst(p.s1);
}

__device__ void shift(int fid, Pa *p) {
    /* fid: fragment id */
    enum {X, Y, Z};
    int d[3]; /* coordinate shift */
    i2shift(fid, d);
    p->s0.x += d[X];
    p->s0.y += d[Y];
    p->s1.x += d[Z];
}
