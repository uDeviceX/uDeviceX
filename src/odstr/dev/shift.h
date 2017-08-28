namespace odstr { namespace sub { namespace dev {

static __device__ void fid2shift(int id, int s[3]) {
    enum {X, Y, Z};
    s[X] = XS * ((id     + 1) % 3 - 1);
    s[Y] = YS * ((id / 3 + 1) % 3 - 1);
    s[Z] = ZS * ((id / 9 + 1) % 3 - 1);
}

__device__ void shiftPart(int s[3], Part *p) {
    enum {X, Y, Z};
    p->r[X] += s[X];
    p->r[Y] += s[Y];
    p->r[Z] += s[Z];
}

__device__ void shift_1p(int i, const int strt[], /*io*/ Part *p) {
    /* i: particle index */
    enum {X, Y, Z};
    int fid;     /* fragment id */
    int shift[3];

    fid  = k_common::fid(strt, i);
    fid2shift(fid, /**/ shift);
    shiftPart(shift, p);
}

__global__ void shift(const int n, const int strt[], /*io*/ float2 *pp) {
    int ws, dw;  /* warp start and shift (lane) */
    Part p; /* [p]article and its [l]ocation in memory */
    Lo l;

    warpco(&ws, &dw); /* warp coordinates */
    if (ws >= n) return;
    pp2Lo(pp, n, ws, /**/ &l);
    readPart(l, /**/ &p);   /* collective */
    if (!endLo(&l, dw))
        shift_1p(ws + dw, strt, /*io*/ &p);
    writePart(&p, /**/ l); /* collective */
}

}}} // namespace
