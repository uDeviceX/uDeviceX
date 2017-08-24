namespace odstr { namespace sub { namespace dev {

struct Pa { /* local particle */
    float2 d0, d1, d2;
};

struct Lo { /* particle [lo]cation in memory
               d: shift in wrap, used for collective access  */
    float2 *p;
    int d;
};

struct Ce { /* coordinates of a cell */
    int ix, iy, iz;
    int id; /* linear index */
};

__device__ void pp2Lo(float2 *pp, int n, int ws, /**/ Lo *l) {
    int dwe; /* warp or buffer end relative to wrap start (`ws') */
    int N_FLOAT2_PER_PARTICLE = 3;
    dwe  = min(warpSize, n - ws);
    l->p = pp + N_FLOAT2_PER_PARTICLE * ws;
    l->d = dwe;
}

__device__ int endLo(Lo *l, int d) { /* is `d' behind the end? */
    /* `d' relative to wrap start */
    return d >= l->d;
}

__device__ void readPa(Lo l, /**/ Pa *p) {
    k_read::AOS6f(l.p, l.d, /**/ p->d0, p->d1, p->d2);
}

__device__ void writePa(Pa *p, /**/ Lo l) {
    k_write::AOS6f(/**/ l.p, l.d, /*i*/ p->d0, p->d1, p->d2);
}

__device__ void shiftPa(int r[3], Pa *p) {
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

__device__ void Pa2Ce(Pa *p, /**/ Ce *c) {
    /* particle to cell coordinates */
    float r[3];
    Pa2r(p, /**/ r);
    r2c(r, /**/ &c->ix, &c->iy, &c->iz, &c->id);
}

__device__ void regCe(Ce *c, int i, /*io*/ int *counts, /**/ uchar4 *subids) {
    /* a particle `i` will lives in `c'. [Reg]ister it. */
    int subindex;
    subindex = atomicAdd(counts + c->id, 1);
    subids[i] = make_uchar4(c->ix, c->iy, c->iz, subindex);
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
    int shift[3];
    Ce c; /* cell coordinates */

    fid  = k_common::fid(strt, i);
    fid2shift(fid, /**/ shift);
    shiftPa(shift, p);

    Pa2Ce(p, /**/ &c); /* to cell coordinates */
    checkPav(p); /* check velocity */
    regCe(&c, i, /*io*/ counts, subids); /* register */
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
