/* particle - float2 union */
union Part {
    float2 f2[3];
    struct { float3 r, v; };
};

/* position - float2 union */
union Pos {
    float2 f2[2];
    struct { float3 r; float dummy; };
};

static __device__ Pos tex2Pos(const Texo<float2> vert, int id) {
    Pos r;
    r.f2[0] = fetch(vert, 3 * id + 0);
    r.f2[1] = fetch(vert, 3 * id + 1);
    return r;
}

static __device__ Part tex2Part(const Texo<float2> vert, int id) {
    Part p;
    p.f2[0] = fetch(vert, 3 * id + 0);
    p.f2[1] = fetch(vert, 3 * id + 1);
    p.f2[2] = fetch(vert, 3 * id + 2);
    return p;
}

static __device__ float3 adj_tris(const Texo<float2> vert,  const Part p0, const float *av,
                                  const Shape0 shape, const Rnd0 rnd,
                                  adj::Map *m) {
    float3 f, fv, fr;
    int i1, i2, rbc;
    float area, volume;
    i1 = m->i1; i2 = m->i2; rbc = m->rbc;

    const Part p1 = tex2Part(vert, i1);
    const Pos  r2 = tex2Pos(vert,  i2);

    area = av[2*rbc]; volume = av[2 * rbc + 1];
    f  = tri(p0.r, p1.r, r2.r, shape, area, volume);
    
    fv = visc(p0.r, p1.r, p0.v, p1.v);
    add(&fv, /**/ &f);

    fr = frnd(p0.r, p1.r, rnd);
    add(&fr, /**/ &f);
    return f;
}

static __device__ float3 adj_dihedrals(const Texo<float2> vert, float3 r0,
                                       const Shape0 shape,
                                       adj::Map *m) {
    Pos r1, r2, r3, r4;
    r1 = tex2Pos(vert, m->i1);
    r2 = tex2Pos(vert, m->i2);
    r3 = tex2Pos(vert, m->i3);
    r4 = tex2Pos(vert, m->i4);
    return dih(r0, r1.r, r2.r, r3.r, r4.r);
}

static __global__ void force(int md, int nv, int nc, const Texo<float2> vert, float *rnd,
                             const Texo<int> adj0, const Texo<int> adj1,
                             const Shape shape,
                             const float *__restrict__ av, /**/ float *ff) {
    assert(md == RBCmd);
    assert(nv == RBCnv);
    int i, pid;
    float3 f, fd;
    adj::Map m;
    Shape0 shape0;
    Rnd0   rnd0;
    int valid;

    i = threadIdx.x + blockDim.x * blockIdx.x;
    pid = i / md;

    if (pid >= nc * nv) return;
    valid = adj::dev(md, nv, i, adj0, adj1, /**/ &m);
    if (!valid) return;
    edg_shape(shape, i % (md * nv),  /**/ &shape0);
    edg_rnd(    rnd, i,              /**/ &rnd0);

    const Part p0 = tex2Part(vert, m.i0);

    f  = adj_tris(vert, p0, av,    shape0, rnd0, &m);
    fd = adj_dihedrals(vert, p0.r, shape0,       &m);
    add(&fd, /**/ &f);

    atomicAdd(&ff[3 * pid + 0], f.x);
    atomicAdd(&ff[3 * pid + 1], f.y);
    atomicAdd(&ff[3 * pid + 2], f.z);
}
