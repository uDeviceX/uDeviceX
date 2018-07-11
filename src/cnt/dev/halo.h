template <typename Par>
static __device__
void interactions_one_p(Par params, int3 L, float seed, PairPa a, int aid,
                        const int *starts, const uint *ids_src, const float2 *pp_src,
                        /**/ float *ff_src, float *fA) {
    enum {X, Y, Z};
    Map m;
    int mapstatus;
    PairPa b;
    float fx, fy, fz;
    float xforce = 0, yforce = 0, zforce = 0;
    int zplane;
    int i, slot, oid, bid, sentry;
    uint code;
    float rnd;

    for (zplane = 0; zplane < 3; ++zplane) {
        mapstatus = tex2map(L, zplane, a.x, a.y, a.z, starts, /**/ &m);
        if (mapstatus == EMPTY) continue;
        for (i = 0; !endp(m, i); ++i) {
            slot = m2id(m, i);
            code = ids_src[slot];
            clist_decode_id(code, /**/ &oid, &bid);
            
            fetch(bid, pp_src, /**/ &b);
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
    }
    fA[X] += xforce; fA[Y] += yforce; fA[Z] += zforce;
}

template <typename Par>
__global__ void halo(Par params, int3 L, float seed,
                     const int27 starts, int n_dst, Pap26 hpp,
                     const int *cellstarts, const uint *ids_src, const float2 *pp_src,
                     /**/ Fop26 hff, float *ff_src) {
    int aid, start;
    int fid;
    PairPa a;
    float *fA;    
    
    aid = threadIdx.x + blockDim.x * blockIdx.x;
    if (aid >= n_dst) return;

    fid = frag_dev::frag_get_fid(starts.d, aid);
    start = starts.d[fid];
    fetch(aid - start, (const float2*)hpp.d[fid], &a);
    fA = hff.d[fid][aid - start].f;
    interactions_one_p(params, L, seed, a, aid, cellstarts, ids_src, pp_src,
                       /**/ ff_src, fA);
}
