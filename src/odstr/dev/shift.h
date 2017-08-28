namespace odstr { namespace sub { namespace dev {

__device__ void shift_1p(int i, const int strt[], /*io*/ Pa *p) {
    /* i: particle index */
    enum {X, Y, Z};
    int fid;     /* fragment id */
    int shift[3];

    fid  = k_common::fid(strt, i);
    fid2shift(fid, /**/ shift);
    shiftPa(shift, p);
}

__global__ void shift(const int n, const int strt[], /*io*/ float2 *pp) {
    int ws, dw;  /* warp start and shift (lane) */
    Pa p; /* [p]article and its [l]ocation in memory */
    Lo l;

    warpco(&ws, &dw); /* warp coordinates */
    if (ws >= n) return;
    pp2Lo(pp, n, ws, /**/ &l);
    readPa(l, /**/ &p);   /* collective */
    if (!endLo(&l, dw))
        shift_1p(ws + dw, strt, /*io*/ &p);
    writePa(&p, /**/ l); /* collective */
}

}}} // namespace
