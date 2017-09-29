struct Pa { /* local particle */
    float x, y, z;
    float vx, vy, vz;
};

static __device__ void pp2p(Particle *pp, int i, /**/
                          forces::Pa *a) {
    float *r, *v;

    pp += i;
    r = pp->r;
    v = pp->v;
    forces::rvk2p(r, v, SOLID_KIND, /**/ a);
}

__global__ void halo(int n, float seed) {
    enum {X, Y, Z};
    Map m;
    int mapstatus;
    forces::Pa a, b;
    float fx, fy, fz;
    float x, y, z;
    int aid, start;
    int fid;
    float xforce = 0, yforce = 0, zforce = 0;
    int zplane;
    int i;
    int slot;
    int objid, bid;
    int sentry;
    float2 stmp0, stmp1, stmp2;
    float rnd;

    aid = threadIdx.x + blockDim.x * blockIdx.x;
    if (aid >= n) return;

    fid = k_common::fid(h::starts, aid);
    start = h::starts[fid];
    pp2p(h::pp[fid], aid - start, &a);
    forces::p2r3(&a, /**/ &x, &y, &z);

    float *fA;
    fA = h::ff[fid][aid - start].f;
    for (zplane = 0; zplane < 3; ++zplane) {
        mapstatus = tex2map(zplane, x, y, z, /**/ &m);
        if (mapstatus == EMPTY) continue;
        for (i = 0; !endp(m, i); ++i) {
            slot = m2id(m, i);
            get(slot, &objid, &bid);

            sentry = 3 * bid;
            stmp0 = __ldg(c::PP[objid] + sentry);
            stmp1 = __ldg(c::PP[objid] + sentry + 1);
            stmp2 = __ldg(c::PP[objid] + sentry + 2);
            rnd = rnd::mean0var1ii(seed, aid, bid);
            forces::f2k2p(stmp0, stmp1, stmp2, SOLID_KIND, /**/ &b);
            forces::gen(a, b, rnd, &fx, &fy, &fz);
            xforce += fx;
            yforce += fy;
            zforce += fz;
            atomicAdd(c::FF[objid] + sentry,     -fx);
            atomicAdd(c::FF[objid] + sentry + 1, -fy);
            atomicAdd(c::FF[objid] + sentry + 2, -fz);
        }
    }

    fA[X] += xforce; fA[Y] += yforce; fA[Z] += zforce;
}
