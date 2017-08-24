namespace odstr { namespace sub { namespace dev {

__device__ void warpco(/**/ int *ws, int *dw) { /* warp [co]ordinates */
    /* ws: start, dw: shift (lane) */
    int warp;
    warp = threadIdx.x / warpSize;
    *dw   = threadIdx.x % warpSize;
    *ws   = warpSize * warp + blockDim.x * blockIdx.x;
}

struct Pa { /* local particle */
    float2 d0, d1, d2;
};

struct Lo { /* particle [lo]cation in memory
               d: shift in wrap, used for collective access  */
    float2 *p;
    int d;
};

__device__ void pp2Lo(float2 *pp, int n, int ws, /**/ Lo *l) {
    int dwe; /* warp or buffer end relative wrap start */
    dwe  = min(warpSize, n - ws);
    l->p = pp + 3*ws;
    l->d = dwe;
}

__device__ int endLo(Lo *l, int d) { /* is `d' behind the end? */
    return d >= l->d;
}

__device__ void readPa(Lo l, /**/ Pa *p) {
    k_read::AOS6f(l.p, l.d, /**/ p->d0, p->d1, p->d2);
}

__device__ void writePa(Pa *p, /**/ Lo l) {
    k_write::AOS6f(/**/ l.p, l.d, /*i*/ p->d0, p->d1, p->d2);
}

__device__ void shiftPa(float r[3], Pa *p) {
    enum {X, Y, Z};
    p->d0.x += r[X];   p->d0.y += r[Y];   p->d1.x += r[Z];
}

__device__ void Pa2r(Pa *p, /**/ float r[3]) { /* to position */
    enum {X, Y, Z};
    r[X] = p->d0.x;   r[Y] = p->d0.y;   r[Z] = p->d1.x;
}

__device__ void Pa2v(Pa *p, /**/ float v[3]) { /* to velocity */
    enum {X, Y, Z};
    v[X] = p->d1.y;   v[Y] = p->d2.x;   v[Z] = p->d2.y;
}

__device__ void r2c(float r[3], /**/ int* ix, int* iy, int* iz, int* i) {
    /* position to cell coordinates */
    enum {X, Y, Z};
    int x, y, z;
    x = x2c(r[X], XS);
    y = x2c(r[Y], YS);
    z = x2c(r[Z], ZS);
    *i  = x + XS * (y + YS * z);

    *ix = x; *iy = y; *iz = z;
}

__device__ void Pa2c(Pa *p, /**/ int* ix, int* iy, int* iz, int* i) {
    /* particle to cell coordinates */
    float r[3];
    Pa2r(p, /**/ r);
    r2c(r, /**/ ix, iy, iz, i);
}

__device__ void checkPav(Pa *p) { /* check particle velocity */
    enum {X, Y, Z};
    float v[3];
    Pa2v(p, /**/ v);
    check_vel(v[X], XS);
    check_vel(v[Y], YS);
    check_vel(v[Z], ZS);
}

__device__ void subindex0(int i, const int strt[], /*io*/ Pa *p, int *counts, /**/ uchar4 *subids) {
    /* i: particle index */
    enum {X, Y, Z};
    int fid;     /* fragment id */
    int xi, yi, zi, cid, subindex;
    float shift[3], r[3];

    fid  = k_common::fid(strt, i);
    fid2shift(fid, /**/ shift);
    shiftPa(shift, p);

    Pa2c(p, /**/ *xi, *yi, *zi, *cid); /* to cell coordinates */
    checkPav(p); /* check velocity */

    subindex = atomicAdd(counts + cid, 1);
    subids[i] = make_uchar4(xi, yi, zi, subindex);
}

__global__ void subindex(const int n, const int strt[], /*io*/ float2 *pp, int *counts, /**/ uchar4 *subids) {
    enum {X, Y, Z};
    int ws, dw;  /* warp start and shift (lane) */
    Pa p; /* [p]article and its [l]ocation in memory */
    Lo l;

    warpco(&ws, &dw); /* warp coordinates */
    if (ws >= n) return;
    pp2Lo(pp, n, ws, /**/ &l);
    readPa(l, /**/ &p);   /* collective */
    if (!endLo(&l, dw))
        subindex0(ws + dw, strt, /*io*/ &p, counts, /**/ subids);
    writePa(&p, /**/ l); /* collective */
}

}}} /* namespace */
