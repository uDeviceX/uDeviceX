template <typename Wvel_v>
static __device__ void fetch_wall(Wvel_v wv, Coords_v c, Texo<float4> pp, int i, /**/ PairPa *a) {
    float3 r, v; /* wall velocity */
    float4 r0;
    r0 = texo_fetch(pp, i);
    r = make_float3(r0.x, r0.y, r0.z);
    wvel(wv, c, r, /**/ &v);

    a->x  = r.x;  a->y  = r.y;   a->z  = r.z;
    a->vx = v.x;  a->vy = v.y;   a->vz = v.z;
}

template <typename Par, typename Wvel_v, typename Fo>
static __device__ void force0(Par params, Wvel_v wv, Coords_v c, PairPa a, int aid, int zplane,
                              float seed, WallForce wa, /**/ Fo *fa) {
    map::Map m;
    PairPa b;  /* wall particles */
    Fo f;
    float rnd;
    int i, bid;

    if (sdf_far(&wa.sdf_v, a.x, a.y, a.z)) return;

    map::ini(wa.L, zplane, wa.start, wa.n, a.x, a.y, a.z, /**/ &m);

    for (i = 0; !map::endp(m, i); ++i) {
        bid = map::m2id(m, i);
        fetch_wall(wv, c, wa.pp, bid, /**/ &b);
        rnd = rnd::mean0var1ii(seed, aid, bid);
        pair_force(&params, a, b, rnd, /**/ &f);
        pair_add(&f, fa);
    }
}

template <typename Par, typename Wvel_v, typename Parray, typename Farray>
__global__ void force(Par params, Wvel_v wv, Coords_v c, Parray parray, int np, float seed, WallForce wa, /**/ Farray farray) {
    PairPa a; /* bulk particle */
    int gid, aid, zplane;
    auto ftot = farray_fo0(farray);
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    aid    = gid / 3;
    zplane = gid % 3;

    if (aid >= np) return;
    parray_get(parray, aid, /**/ &a);

    force0(params, wv, c, a, aid, zplane, seed, wa, /**/ &ftot);

    farray_atomic_add<1>(&ftot, aid, /**/ farray);
}

static __device__ float rep_kernel(float l, float x) {
    float arg = l * (x + 1.f);
    return expf(arg) - 1.f;
}

static __device__ void repulse_one_p(const float3 *r, const Sdf_v *sdf_v, float3 *f) {
    float s, f0;
    float3 g;
    s = sdf_eval(sdf_v, r->x, r->y, r->z);
    g = sdf_ugrad(sdf_v, r);
    f0 = - rep_kernel(3.f, s);
    f->x = f0 * g.x;
    f->y = f0 * g.y;
    f->z = f0 * g.z;
}

static __device__ float3 get_pos(const Particle *p) {
    enum {X, Y, Z};
    return make_float3(p->r[X], p->r[Y], p->r[Z]);
}

__global__ void repulse(int n, const Particle *pp, Sdf_v sdf_v, /**/ Force *ff) {
    enum {X, Y, Z};
    int pid;
    float3 r, f;
    pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid >= n) return;
    r = get_pos(pp + pid);
    
    if (sdf_far(&sdf_v, r.x, r.y, r.z)) return;

    repulse_one_p(&r, &sdf_v, &f);

    ff[pid].f[X] += f.x;
    ff[pid].f[Y] += f.y;
    ff[pid].f[Z] += f.z;
}
