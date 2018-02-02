struct Part { real3 r, v; };
struct Pos  { real3 r; };


static __device__ Pos tex2Pos(const Particle *pp, int i) {
    enum {X, Y, Z};
    Pos r;
    r.r.x = pp[i].r[X];
    r.r.y = pp[i].r[Y];
    r.r.z = pp[i].r[Z];
    return r;
}

static __device__ Part tex2Part(const Particle *pp, int i) {
    enum {X, Y, Z};
    Part p;
    const real *r, *v;
    r = pp[i].r; v = pp[i].v;
    p.r.x = r[X]; p.r.y = r[Y]; p.r.z = r[Z];
    p.v.x = v[X]; p.v.y = v[Y]; p.v.z = v[Z];
    return p;
}

static __device__ real3 adj_tris(float dt0,
                                 RbcParams_v par, const Particle *pp,  const Part p0, const real *av,
                                 const Shape0 shape, const Rnd0 rnd,
                                 AdjMap *m) {
    real3 f, fv, fr;
    int i1, i2, rbc;
    real area, volume;
    i1 = m->i1; i2 = m->i2; rbc = m->rbc;

    const Part p1 = tex2Part(pp, i1);
    const Pos  r2 = tex2Pos(pp,  i2);

    area = av[2*rbc]; volume = av[2 * rbc + 1];
    f  = tri(par, p0.r, p1.r, r2.r, shape, area, volume);
    
    fv = visc(par, p0.r, p1.r, p0.v, p1.v);
    add(&fv, /**/ &f);

    fr = frnd(dt0, par, p0.r, p1.r, rnd);
    add(&fr, /**/ &f);
    return f;
}

static __device__ real3 adj_dihedrals(RbcParams_v par, const Particle *pp, real3 r0,
                                      const Shape0 shape,
                                      AdjMap *m) {
    Pos r1, r2, r3, r4;
    r1 = tex2Pos(pp, m->i1);
    r2 = tex2Pos(pp, m->i2);
    r3 = tex2Pos(pp, m->i3);
    r4 = tex2Pos(pp, m->i4);
    return dih(par, r0, r1.r, r2.r, r3.r, r4.r);
}

__global__ void force(float dt0,
                      RbcParams_v par, int md, int nv, int nc, const Particle *pp, real *rnd,
                      Adj_v adj,
                      const Shape shape,
                      const real *av, /**/ float *ff) {
    int i, pid;
    real3 f, fd;
    AdjMap m;
    Shape0 shape0;
    Rnd0   rnd0;
    int valid;

    i = threadIdx.x + blockDim.x * blockIdx.x;
    pid = i / md;

    if (pid >= nc * nv) return;
    valid = adj_get_map(i, &adj, /**/ &m);
    if (!valid) return;
    edg_shape(shape, i % (md * nv),         /**/ &shape0);
    edg_rnd(  shape, i % (md * nv), rnd, i, /**/ &rnd0);

    const Part p0 = tex2Part(pp, m.i0);

    f  = adj_tris(dt0, par, pp, p0, av,    shape0, rnd0, &m);
    fd = adj_dihedrals(par, pp, p0.r, shape0,       &m);
    add(&fd, /**/ &f);

    atomicAdd(&ff[3 * pid + 0], f.x);
    atomicAdd(&ff[3 * pid + 1], f.y);
    atomicAdd(&ff[3 * pid + 2], f.z);
}
