namespace k_rex { /* particle and force packing dependent part : TODO : give a name */
__device__ Pa pp2p(const float2 *pp, int i) {
    /* pp array to a particle */
    Pa p;
    float2 s0, s1, s2;
    pp += 3*i;
    s0 = __ldg(pp++); s1 = __ldg(pp++); s2 = __ldg(pp++);
    p.s0 = s0; p.s1 = s1; p.s2 = s2;
    return p;
}

__device__ void p2pp(Pa p, int n, int i, /**/ float2 *pp) {
    /* collective write : p to buffer pp */
    k_write::AOS6f(pp + 3 * i, n, p.s0, p.s1, p.s2);
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
