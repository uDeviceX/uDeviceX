static __device__ void f22p(float2 a, float2 b, float2 c, PairPa *p) {
    p->x  = a.x; p->y  = a.y; p->z  = b.x;
    p->vx = b.y; p->vy = c.x; p->vz = c.y;
}

static __device__ void fetch(int i, const float2 *pp, PairPa *p) {
    float2 a, b, c;
    a = __ldg(pp + 3 * i + 0);
    b = __ldg(pp + 3 * i + 1);
    c = __ldg(pp + 3 * i + 2);
    f22p(a, b, c, p);
}

template <typename Par>
__global__ void bulk(bool self, Par params, int3 L, int n_dst, const float2 *pp_dst,
                     const int *starts, const uint *ids_src, const float2 *pp_src,
                     float seed, /**/ float *ff_dst, float *ff_src) {
    Map m; /* see map/ */

    float fx, fy, fz, rnd;
    PairPa a, b;
    int i, gid, aid, zplane;
    float xforce, yforce, zforce;
    int mapstatus;
    int slot, objid, bid, sentry;
    uint code;
    
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    aid = gid / 3;
    zplane = gid % 3;

    if (aid >= n_dst) return;

    fetch(aid, pp_dst, &a);
    mapstatus = r2map(L, zplane, n_dst, a.x, a.y, a.z, starts, /**/ &m);
    
    if (mapstatus == EMPTY) return;
    xforce = yforce = zforce = 0;
    for (i = 0; !endp(m, i); ++i) {
        slot = m2id(m, i);
        code = ids_src[slot];
        clist_decode_id(code, /**/ &objid, &bid);

        if (self && aid <= bid) continue;

        fetch(bid, pp_src, &b);
        
        rnd = rnd::mean0var1ii(seed, aid, bid);

        pair(params, a, b, rnd, /**/ &fx, &fy, &fz);

        xforce += fx;
        yforce += fy;
        zforce += fz;

        sentry = 3 * bid;
        atomicAdd(ff_src + sentry,     -fx);
        atomicAdd(ff_src + sentry + 1, -fy);
        atomicAdd(ff_src + sentry + 2, -fz);
    }

    atomicAdd(ff_dst + 3 * aid + 0, xforce);
    atomicAdd(ff_dst + 3 * aid + 1, yforce);
    atomicAdd(ff_dst + 3 * aid + 2, zforce);
}
