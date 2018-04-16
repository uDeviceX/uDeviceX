struct Part { float3 r, v; };
typedef float3 Pos;

static __device__ Pos fetchPos(const Particle *pp, int i) {
    enum {X, Y, Z};
    Pos r;
    r.x = pp[i].r[X];
    r.y = pp[i].r[Y];
    r.z = pp[i].r[Z];
    return r;
}

static __device__ Part fetchPart(const Particle *pp, int i) {
    enum {X, Y, Z};
    Part p;
    const float *r, *v;
    r = pp[i].r; v = pp[i].v;
    p.r.x = r[X]; p.r.y = r[Y]; p.r.z = r[Z];
    p.v.x = v[X]; p.v.y = v[Y]; p.v.z = v[Z];
    return p;
}

template <typename RndInfo>
static __device__ void adj_tris(float dt,
                                const RbcParams_v *par, const Particle *pp,  const Part p0, const float *av,
                                const StressInfo si, RndInfo ri,
                                AdjMap *m, /*io*/ float f[3]) {
    enum {X, Y, Z};
    float3 f0;
    int i1, i2, rbc;
    float area, volume;
    i1 = m->i1; i2 = m->i2; rbc = m->rbc;

    const Part p1 = fetchPart(pp, i1);
    const Pos  r2 = fetchPos(pp,  i2);

    area = av[2*rbc]; volume = av[2 * rbc + 1];
    f0 = ftri(par, p0.r, p1.r, r2, si, area, volume);
    f[X] += f0.x; f[Y] += f0.y; f[Z] += f0.z;
    
    f0 = fvisc(par, p0.r, p1.r, p0.v, p1.v);
    f[X] += f0.x; f[Y] += f0.y; f[Z] += f0.z;

    f0 = frnd(dt, par, p0.r, p1.r, ri);
    f[X] += f0.x; f[Y] += f0.y; f[Z] += f0.z;
}

static __device__ void adj_dihedrals(const RbcParams_v *par, const Particle *pp, float3 r0,
                                     AdjMap *m, /*io*/ float f[3]) {
    enum {X, Y, Z};
    Pos r1, r2, r3, r4;
    float3 f0;
    float phi, kb;
    r1 = fetchPos(pp, m->i1);
    r2 = fetchPos(pp, m->i2);
    r3 = fetchPos(pp, m->i3);
    r4 = fetchPos(pp, m->i4);

    phi = par->phi / 180.0 * M_PI;
    kb  = par->kb;
    
    f0 = force_kantor0_dev::dih_a(phi, kb, r0, r2, r1, r4);
    f[X] += f0.x; f[Y] += f0.y; f[Z] += f0.z;
    
    f0 = force_kantor0_dev::dih_b(phi, kb, r1, r0, r2, r3);
    f[X] += f0.x; f[Y] += f0.y; f[Z] += f0.z;

}

template <typename Stress_v, typename Rnd_v>
__global__ void force(float dt,
                      RbcParams_v par, int md, int nv, int nc, const Particle *pp,
                      Adj_v adj,
                      Stress_v sv, Rnd_v rv,
                      const float *av, /**/ float *ff) {
    enum {X, Y, Z};
    int i, pid, valid;
    float f[3];
    AdjMap m;
    StressInfo si;

    i = threadIdx.x + blockDim.x * blockIdx.x;
    pid = i / md;

    if (pid >= nc * nv) return;
    valid = adj_get_map(i, &adj, /**/ &m);
    if (!valid) return;
    si = fetch_stress_info(i % (md * nv), sv);
    auto ri = fetch_rnd_info(i % (md * nv), i, rv);

    const Part p0 = fetchPart(pp, m.i0);

    f[X] = f[Y] = f[Z] = 0;
    adj_tris(dt, &par, pp, p0, av, si, ri, &m, /*io*/ f);
    adj_dihedrals(&par, pp, p0.r, &m,          /*io*/ f);

    atomicAdd(&ff[3 * pid + 0], f[X]);
    atomicAdd(&ff[3 * pid + 1], f[Y]);
    atomicAdd(&ff[3 * pid + 2], f[Z]);
}
