namespace k_rex { /* particle and force packing dependent part : TODO : give a name */
__device__ Pa pp2p_col(const float2 *pp, int n, int i) {
    /* pp: array to a particle : [coll]ective */
    Pa p;
    k_read::AOS6f(pp + 3*i, n, /**/ p.s0, p.s1, p.s2);
    return p;
}

__device__ Pa pp2p(const float2 *pp, int i) {
    /* pp array to a particle */
    Pa p;
    pp += 3*i;
    p.s0 = __ldg(pp++); p.s1 = __ldg(pp++); p.s2 = __ldg(pp++);
    return p;
}

__device__ void p2pp(Pa p, int n, int i, /**/ float2 *pp) {
    /* collective write : p to buffer pp */
    k_write::AOS6f(pp + 3*i, n, p.s0, p.s1, p.s2);
}

__device__ void p2xyz(const Pa p, /**/ float *x, float *y, float *z) {
    *x = fst(p.s0);
    *y = scn(p.s0);
    *z = fst(p.s1);
}

__device__ void shift(int fid, Pa *p) {
    /* fid: fragment id */
    enum {X, Y, Z};
    float d[3]; /* coordinate shift */
    fid2dr(fid, d);
    p->s0.x += d[X];
    p->s0.y += d[Y];
    p->s1.x += d[Z];
}
}
