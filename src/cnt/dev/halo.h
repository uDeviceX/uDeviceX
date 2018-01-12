static __device__ void pp2p(Particle *pp, int i, /**/ forces::Pa *a) {
    float *r, *v;
    pp += i;
    r = pp->r;
    v = pp->v;
    forces::rvk2p(r, v, SOLID_KIND, /**/ a);
}

static __device__ void fetch_b(const float2 *pp, int i, /**/ forces::Pa *b) {
    float2 s0, s1, s2;
    pp += 3*i;
    s0 = __ldg(pp++);
    s1 = __ldg(pp++);
    s2 = __ldg(pp  );
    forces::f2k2p(s0, s1, s2, SOLID_KIND, /**/ b);
}

__device__ void halo0(const int *cellstarts, const uint *ids, const float2pWraps lpp, forces::Pa a, int aid, float seed,
                      /**/ ForcepWraps lff, float *fA) {
    enum {X, Y, Z};
    Map m;
    int mapstatus;
    forces::Pa b;
    float fx, fy, fz;
    float x, y, z;
    float xforce = 0, yforce = 0, zforce = 0;
    int zplane;
    int i;
    int slot;
    int objid, bid, sentry;
    float rnd;
    uint code;

    forces::p2r3(&a, /**/ &x, &y, &z);
    for (zplane = 0; zplane < 3; ++zplane) {
        mapstatus = tex2map(zplane, x, y, z, cellstarts, /**/ &m);
        if (mapstatus == EMPTY) continue;
        for (i = 0; !endp(m, i); ++i) {
            slot = m2id(m, i);
            code = ids[slot];
            clist_decode_id(code, &objid, &bid);
            fetch_b(lpp.d[objid], bid, /**/ &b);
            rnd = rnd::mean0var1ii(seed, aid, bid);
            pair(a, b, rnd, /**/ &fx, &fy, &fz);
            xforce += fx;
            yforce += fy;
            zforce += fz;
            
            sentry = 3 * bid;
            atomicAdd(lff.d[objid] + sentry,     -fx);
            atomicAdd(lff.d[objid] + sentry + 1, -fy);
            atomicAdd(lff.d[objid] + sentry + 2, -fz);
        }
    }
    fA[X] += xforce; fA[Y] += yforce; fA[Z] += zforce;
}

__global__ void halo(const int *cellstarts, const uint *ids, float seed, const int27 starts, const float2pWraps lpp,
                     int n, Pap26 hpp, /**/ ForcepWraps lff, Fop26 hff) {
    int aid, start;
    int fid;
    forces::Pa a;
    float *fA;    
    
    aid = threadIdx.x + blockDim.x * blockIdx.x;
    if (aid >= n) return;

    fid = frag_get_fid(starts.d, aid);
    start = starts.d[fid];
    pp2p(hpp.d[fid], aid - start, &a);
    fA = hff.d[fid][aid - start].f;
    halo0(cellstarts, ids, lpp, a, aid, seed, /**/ lff, fA);
}
