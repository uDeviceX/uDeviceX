namespace odstr { namespace sub { namespace dev {

struct Ce { /* coordinates of a cell */
    int ix, iy, iz;
    int id; /* linear index */
};

static __device__ void Part2r(const Part *p, /**/ float *r) {
    enum {X, Y, Z};
    r[X] = fst(p->d0);
    r[Y] = scn(p->d0);
    r[Z] = fst(p->d1);
}

static  __device__ void Part2v(const Part *p, /**/ float *v) {
    enum {X, Y, Z};
    v[X] = scn(p->d1);
    v[Y] = fst(p->d2);
    v[Z] = scn(p->d2);
}

static __device__ void Part2Ce(const Part *p, /**/ Ce *c) {
    /* particle to cell coordinates */
    float r[3];
    Part2r(p, r);
    r2c(r, /**/ &c->ix, &c->iy, &c->iz, &c->id);
}

static __device__ void regCe(Ce *c, int i, /*io*/ int *counts, /**/ uchar4 *subids) {
    /* a particle `i` will lives in `c'. [Reg]ister it. */
    int subindex;
    subindex = atomicAdd(counts + c->id, 1);
    subids[i] = make_uchar4(c->ix, c->iy, c->iz, subindex);
}

static __device__ void checkPav(const Part *p) { /* check particle velocity */
    enum {X, Y, Z};
    float v[3];
    Part2v(p, v);
    check_vel(v[X], XS);
    check_vel(v[Y], YS);
    check_vel(v[Z], ZS);
}

static __device__ void subindex0(int i, const int strt[], const Part *p, /*io*/ int *counts, /**/ uchar4 *subids) {
    /* i: particle index */
    enum {X, Y, Z};
    Ce c; /* cell coordinates */

    Part2Ce(p, /**/ &c); /* to cell coordinates */
    checkPav(p); /* check velocity */
    regCe(&c, i, /*io*/ counts, subids); /* register */
}

__global__ void subindex(const int n, const int strt[], float2 *pp, /*io*/ int *counts, /**/ uchar4 *subids) {
    enum {X, Y, Z};
    int ws, dw;  /* warp start and shift (lane) */
    Part p; /* [p]article and its [l]ocation in memory */
    Lo l;

    warpco(&ws, &dw); /* warp coordinates */
    if (ws >= n) return;
    pp2Lo(pp, n, ws, /**/ &l);
    readPart(l, /**/ &p);   /* collective */
    if (!endLo(&l, dw))
        subindex0(ws + dw, strt, /*io*/ &p, counts, /**/ subids);
}

}}} /* namespace */
