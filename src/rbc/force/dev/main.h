struct Part { real3 r, v; };
typedef real3 Pos;


static __device__ Pos tex2Pos(const Particle *pp, int i) {
    enum {X, Y, Z};
    Pos r;
    r.x = pp[i].r[X];
    r.y = pp[i].r[Y];
    r.z = pp[i].r[Z];
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

template <typename RndInfo>
static __device__ real3 adj_tris(real dt,
                                 RbcParams_v par, const Particle *pp,  const Part p0, const real *av,
                                 const StressInfo si, RndInfo ri,
                                 AdjMap *m) {
    real3 f, fv, fr;
    int i1, i2, rbc;
    real area, volume;
    i1 = m->i1; i2 = m->i2; rbc = m->rbc;

    const Part p1 = tex2Part(pp, i1);
    const Pos  r2 = tex2Pos(pp,  i2);

    area = av[2*rbc]; volume = av[2 * rbc + 1];
    f  = ftri(par, p0.r, p1.r, r2, si, area, volume);
    
    fv = fvisc(par, p0.r, p1.r, p0.v, p1.v);
    add(&fv, /**/ &f);

    fr = frnd(dt, par, p0.r, p1.r, ri);
    add(&fr, /**/ &f);
    return f;
}

static __device__ real3 adj_dihedrals(RbcParams_v par, const Particle *pp, real3 r0,
                                      AdjMap *m) {
    Pos r1, r2, r3, r4;
    real3 f1, f2;
    r1 = tex2Pos(pp, m->i1);
    r2 = tex2Pos(pp, m->i2);
    r3 = tex2Pos(pp, m->i3);
    r4 = tex2Pos(pp, m->i4);

    f1 = fdih<1>(par, r0, r2, r1, r4);
    f2 = fdih<2>(par, r1, r0, r2, r3);
    add(&f1, /**/ &f2);
    return f2;

}

template <typename Stress_v, typename Rnd_v>
__global__ void force(float dt,
                      RbcParams_v par, int md, int nv, int nc, const Particle *pp,
                      Adj_v adj,
                      Stress_v sv, Rnd_v rv,
                      const real *av, /**/ float *ff) {
    int i, pid;
    real3 f, fd;
    AdjMap m;
    StressInfo si;
    int valid;

    i = threadIdx.x + blockDim.x * blockIdx.x;
    pid = i / md;

    if (pid >= nc * nv) return;
    valid = adj_get_map(i, &adj, /**/ &m);
    if (!valid) return;
    si = fetch_stress_info(i % (md * nv), sv);
    auto ri = fetch_rnd_info(i % (md * nv), i, rv);

    const Part p0 = tex2Part(pp, m.i0);

    f  = adj_tris(dt, par, pp, p0, av, si, ri, &m);
    fd = adj_dihedrals(par, pp, p0.r, &m);
    add(&fd, /**/ &f);

    atomicAdd(&ff[3 * pid + 0], f.x);
    atomicAdd(&ff[3 * pid + 1], f.y);
    atomicAdd(&ff[3 * pid + 2], f.z);
}
