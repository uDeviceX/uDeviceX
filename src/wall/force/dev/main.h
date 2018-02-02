template<typename Par>
static __device__ void fetch_wall(Wvel_v wv, Coords_v c, Texo<float4> pp, int i, /**/ PairPa *a) {
    float3 r, v; /* wall velocity */
    float4 r0;
    r0 = fetch(pp, i);
    r = make_float3(r0.x, r0.y, r0.z);
    wvel(wv, c, r, /**/ &v);

    a->x  = r.x;  a->y  = r.y;   a->z  = r.z;
    a->vx = v.x;  a->vy = v.y;   a->vz = v.z;
}

template<typename Par>
static __device__ void force0(Par params, Wvel_v wv, Coords_v c, PairPa a, int aid, int zplane,
                              float seed, WallForce wa, /**/ float *ff) {
    map::Map m;
    PairPa b;  /* wall particles */
    float rnd;
    PairFo f;
    int i, bid;

    if (sdf_far(&wa.sdf_v, a.x, a.y, a.z)) return;

    map::ini(wa.L, zplane, wa.start, wa.n, a.x, a.y, a.z, /**/ &m);
    float xforce = 0, yforce = 0, zforce = 0;
    for (i = 0; !map::endp(m, i); ++i) {
        bid = map::m2id(m, i);
        fetch_wall<Par>(wv, c, wa.pp, bid, /**/ &b);
        rnd = rnd::mean0var1ii(seed, aid, bid);
        pair_force(params, a, b, rnd, /**/ &f);
        xforce += f.x; yforce += f.y; zforce += f.z;
    }
    atomicAdd(ff + 3 * aid + 0, xforce);
    atomicAdd(ff + 3 * aid + 1, yforce);
    atomicAdd(ff + 3 * aid + 2, zforce);
}

template<typename Par>
__global__ void force(Par params, Wvel_v wv, Coords_v c, Cloud cloud, int np, float seed, WallForce wa, /**/ float *ff) {
    PairPa a; /* bulk particle */
    int gid, aid, zplane;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    aid    = gid / 3;
    zplane = gid % 3;

    if (aid >= np) return;
    fetch_p(params, cloud, aid, /**/ &a);

    force0(params, wv, c, a, aid, zplane, seed, wa, /**/ ff);
}
