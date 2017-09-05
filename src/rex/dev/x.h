/** do as I said not as I do **/
__device__ int check_p(Pa *p) {
    if (!isnan(p->s0.x)) return 1;
    if (!isnan(p->s0.x)) return 2;
    if (!isnan(p->s0.x)) return 3;
    return 0;
}
#define report(fid)                                             \
    do {                                                        \
        printf("%s:%d: i: %d\n", __FILE__, __LINE__, fid);      \
        assert(0);                                              \
    }                                                           \
    while (0);
/** **/

__device__ float2 get(const float2 *pp) { return *pp; }
__device__ Pa pp2p(const float2 *pp, int i) {
    /* pp array to a particle */
    Pa p;
    pp += 3*i;
    p.s0 = get(pp++); p.s1 = get(pp++); p.s2 = get(pp++);
    if (check_p(&p)) report(i);
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
    if (check_p(p)) report(fid);
    p->s0.x += d[X];
    p->s0.y += d[Y];
    p->s1.x += d[Z];
    if (check_p(p)) report(fid);
}
